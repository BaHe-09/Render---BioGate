from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from keras_facenet import FaceNet
from typing import List
from sqlalchemy.orm import Session
from database import get_db
from sqlalchemy import text
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="FaceNet Embeddings API",
    description="API para extraer y comparar embeddings faciales usando Keras-FaceNet y PostgreSQL con pgvector",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplaza con tu dominio de la app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el modelo FaceNet
facenet = FaceNet()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convierte la imagen al formato requerido por FaceNet"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def search_similar_embeddings(db: Session, embedding: List[float], threshold: float = 0.7):
    """Busca embeddings similares en la base de datos"""
    try:
        # Convertir a array PostgreSQL
        embedding_array = "[" + ",".join(map(str, embedding)) + "]"
        
        query = text("""
            SELECT v.id_vector, p.id_persona, p.nombre, p.apellido_paterno, 
                   1 - (v.vector <=> :embedding) as similitud
            FROM vectores_identificacion v
            JOIN personas p ON v.id_persona = p.id_persona
            WHERE 1 - (v.vector <=> :embedding) > :threshold
            ORDER BY similitud DESC
            LIMIT 5
        """)
        
        result = db.execute(query, {"embedding": embedding_array, "threshold": threshold})
        return result.fetchall()
        
    except Exception as e:
        logger.error(f"Error en búsqueda de embeddings: {str(e)}")
        raise

@app.post("/get_embeddings", response_model=dict)
async def get_face_embeddings(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint para obtener embeddings faciales y comparar con la base de datos
    
    Parámetros:
    - file: Archivo de imagen que contiene un rostro
    
    Retorna:
    - embeddings: Vector de 512 dimensiones con las características faciales
    - matches: Posibles coincidencias en la base de datos
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer y procesar imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_array = preprocess_image(img)
        
        # Obtener embeddings
        detections = facenet.embeddings([img_array])
        
        if len(detections) == 0:
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")
        
        embedding = detections[0].tolist()
        
        # Buscar coincidencias en la BD
        matches = search_similar_embeddings(db, embedding)
        
        # Formatear resultados
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                "id_persona": match.id_persona,
                "nombre_completo": f"{match.nombre} {match.apellido_paterno}",
                "similitud": float(match.similitud),
                "id_vector": match.id_vector
            })
        
        return {
            "message": "Embeddings generados correctamente",
            "embedding_size": len(embedding),
            "matches": formatted_matches,
            "best_match": formatted_matches[0] if formatted_matches else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error al procesar la imagen")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/", include_in_schema=False)
def health_check():
    return {"status": "ok", "message": "API de reconocimiento facial operativa"}
