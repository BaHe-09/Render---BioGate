import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desactiva GPU

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from numpy.linalg import norm
from typing import List, Optional
from pydantic import BaseModel
from database import get_db
import asyncio
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API",
              description="API para reconocimiento facial usando YOLOv8 y FaceNet")

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
SIMILARITY_THRESHOLD = 0.7  # Umbral de similitud para reconocimiento
MAX_IMAGE_SIZE = 640  # Tamaño máximo para procesamiento

# Modelos (carga diferida)
MODELS = {}
MODELS_READY = False

class FaceRecognitionResult(BaseModel):
    face_id: int
    person_id: Optional[int]
    name: Optional[str]
    similarity: Optional[float]
    is_known: bool
    bounding_box: List[int]

@app.on_event("startup")
async def load_models():
    """Carga los modelos de IA al iniciar la aplicación"""
    global MODELS_READY
    try:
        logger.info("Cargando modelos de IA...")
        
        # Carga YOLO para detección de rostros
        MODELS['yolo'] = YOLO('yolov8n-face-lindevs.pt').to('cpu')
        
        # Carga FaceNet optimizado para embeddings
        MODELS['facenet'] = tf.keras.models.load_model(
            'facenet_keras.h5',
            compile=False
        )
        # Reducimos el modelo para ahorrar memoria
        MODELS['facenet']._layers = MODELS['facenet'].layers[:4]
        
        MODELS_READY = True
        logger.info("✅ Modelos cargados exitosamente")
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {str(e)}")
        raise RuntimeError(f"No se pudieron cargar los modelos: {str(e)}")

def get_yolo_model():
    """Obtiene el modelo YOLO (carga diferida)"""
    if not MODELS_READY:
        raise HTTPException(status_code=503, detail="Modelos no cargados aún")
    return MODELS['yolo']

def get_facenet_model():
    """Obtiene el modelo FaceNet (carga diferida)"""
    if not MODELS_READY:
        raise HTTPException(status_code=503, detail="Modelos no cargados aún")
    return MODELS['facenet']

@app.get("/health")
async def health_check():
    """Endpoint de salud para Render"""
    return {
        "status": "ready" if MODELS_READY else "loading",
        "service": "face-recognition",
        "models_loaded": MODELS_READY
    }

@app.post("/recognize-faces")
async def recognize_faces(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    threshold: float = SIMILARITY_THRESHOLD
):
    """
    Endpoint para reconocimiento facial.
    Recibe una imagen y devuelve los rostros reconocidos.
    """
    try:
        if not MODELS_READY:
            raise HTTPException(status_code=503, detail="Modelos no cargados aún")

        # 1. Leer y redimensionar imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Reducción de tamaño para ahorrar memoria
        img = cv2.resize(img, (MAX_IMAGE_SIZE, int(MAX_IMAGE_SIZE * img.shape[0]/img.shape[1])))
        
        # 2. Detección de rostros con YOLO
        yolo = get_yolo_model()
        results = yolo(img, verbose=False)
        
        recognition_results = []
        for i, det in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, det[:4])
            
            # 3. Preprocesamiento para FaceNet
            face_img = img[y1:y2, x1:x2]
            face_resized = cv2.resize(face_img, (160, 160))
            face_normalized = (face_resized - 127.5) / 128.0
            
            # 4. Generar embedding
            facenet = get_facenet_model()
            embedding = facenet.predict(np.expand_dims(face_normalized, axis=0))[0]
            embedding /= norm(embedding)
            
            # 5. Buscar coincidencia en DB
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en reconocimiento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def find_closest_match(db: Session, embedding: np.ndarray, threshold: float):
    """Busca coincidencias en la base de datos"""
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

@app.get("/")
async def root():
    """Endpoint raíz con documentación básica"""
    return {
        "message": "Bienvenido a la API de reconocimiento facial",
        "status": "active",
        "endpoints": {
            "recognize": "POST /recognize-faces",
            "health": "GET /health",
            "docs": "/docs"
        },
        "models_ready": MODELS_READY
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=120
    )
