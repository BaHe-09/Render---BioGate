import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import get_db
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from ultralytics import YOLO
from numpy.linalg import norm
import io

app = FastAPI()

# Modelos
yolo_model = YOLO('yolov8n-face-lindevs.pt')
facenet_model = tf.keras.models.load_model('facenet_keras.h5')  # Descarga previamente este modelo

# Configuración
SIMILARITY_THRESHOLD = 0.7  # Ajusta según tus necesidades

class FaceMatchResult(BaseModel):
    person_id: int
    name: str
    similarity: float
    is_known: bool

@app.post("/detect-and-recognize")
async def detect_and_recognize(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    threshold: float = SIMILARITY_THRESHOLD
):
    try:
        # 1. Detección de rostros con YOLO
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar rostros
        results = yolo_model(img)
        faces = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = img[y1:y2, x1:x2]
                faces.append(face_img)
        
        if not faces:
            return JSONResponse(content={"message": "No faces detected"})
        
        # 2. Procesar cada rostro con FaceNet
        recognition_results = []
        for face_img in faces:
            # Preprocesamiento para FaceNet
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = (face_resized - 127.5) / 128.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            
            # Generar embedding
            embedding = facenet_model.predict(face_expanded)[0]
            embedding_normalized = embedding / norm(embedding)
            
            # 3. Buscar en la base de datos
            closest_match = find_closest_match(db, embedding_normalized, threshold)
            
            if closest_match:
                recognition_results.append({
                    "person_id": closest_match["id_persona"],
                    "name": closest_match["nombre"],
                    "similarity": float(1 - closest_match["distance"]),
                    "is_known": True
                })
            else:
                recognition_results.append({
                    "person_id": None,
                    "name": "Unknown",
                    "similarity": 0,
                    "is_known": False
                })
        
        return {"faces": recognition_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_closest_match(db: Session, embedding: np.ndarray, threshold: float):
    embedding_list = embedding.tolist()
    
    query = """
        SELECT 
            v.id_vector,
            v.id_persona,
            p.nombre,
            p.apellido_paterno,
            p.correo_electronico,
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

# Health check para Render (evita hibernación)
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "face-api"}
