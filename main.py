import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desactiva GPU

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from numpy.linalg import norm
from typing import List, Optional
from pydantic import BaseModel
from database import get_db

app = FastAPI()

# Configuración
SIMILARITY_THRESHOLD = 0.7  # Umbral de similitud para reconocimiento
MAX_IMAGE_SIZE = 640  # Tamaño máximo para procesamiento en Render free tier

# Modelos (carga diferida)
MODELS = {}

def get_yolo_model():
    if 'yolo' not in MODELS:
        MODELS['yolo'] = YOLO('yolov8n-face-lindevs.pt').to('cpu')
    return MODELS['yolo']

def get_facenet_model():
    if 'facenet' not in MODELS:
        # Versión ligera de FaceNet para Render free tier
        model = tf.keras.models.load_model('facenet_keras.h5', compile=False)
        model._layers = model.layers[:4]  # Usar solo primeras capas para reducir memoria
        MODELS['facenet'] = model
    return MODELS['facenet']

class FaceRecognitionResult(BaseModel):
    face_id: int
    person_id: Optional[int]
    name: Optional[str]
    similarity: Optional[float]
    is_known: bool
    bounding_box: List[int]

@app.post("/recognize-faces")
async def recognize_faces(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    threshold: float = SIMILARITY_THRESHOLD
):
    try:
        # 1. Leer y redimensionar imagen para ahorrar memoria
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (MAX_IMAGE_SIZE, int(MAX_IMAGE_SIZE * img.shape[0]/img.shape[1])))
        
        # 2. Detección de rostros con YOLO
        yolo = get_yolo_model()
        results = yolo(img, verbose=False)  # verbose=False para reducir logs
        
        recognition_results = []
        for i, det in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, det[:4])
            
            # 3. Recortar rostro y preprocesar para FaceNet
            face_img = img[y1:y2, x1:x2]
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = (face_resized - 127.5) / 128.0
            
            # 4. Generar embedding con FaceNet (versión ligera)
            facenet = get_facenet_model()
            embedding = facenet.predict(np.expand_dims(face_normalized, axis=0))[0]
            embedding /= norm(embedding)  # Normalizar
            
            # 5. Buscar coincidencia en la base de datos
            match = find_closest_match(db, embedding, threshold)
            
            recognition_results.append(FaceRecognitionResult(
                face_id=i,
                person_id=match["id_persona"] if match else None,
                name=match["nombre"] if match else "Desconocido",
                similarity=float(1 - match["distance"]) if match else 0,
                is_known=match is not None,
                bounding_box=[x1, y1, x2, y2]
            ))
        
        return {"results": recognition_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_closest_match(db: Session, embedding: np.ndarray, threshold: float):
    """Busca el vector más cercano en la base de datos"""
    embedding_list = embedding.tolist()
    
    query = """
        SELECT 
            v.id_vector,
            v.id_persona,
            p.nombre,
            p.apellido_paterno,
            v.vector <-> %s AS distance
        FROM 
            vectores_identificacion v
        JOIN 
            personas p ON v.id_persona = p.id_persona
        WHERE 
            p.activo = TRUE
        ORDER BY 
            distance ASC
        LIMIT 1
    """
    
    result = db.execute(query, (embedding_list,)).fetchone()
    
    if result and result["distance"] <= threshold:
        return {
            "id_vector": result["id_vector"],
            "id_persona": result["id_persona"],
            "nombre": f"{result['nombre']} {result['apellido_paterno']}",
            "distance": result["distance"]
        }
    return None

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "memory_usage": f"{os.getpid()} MB"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Usa $PORT o 10000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
