from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from keras_facenet import FaceNet
from typing import List, Optional
from sqlalchemy.orm import Session
from database import get_db
from sqlalchemy import text
import logging
from datetime import datetime, timezone
import pytz

app = FastAPI(
    title="FaceNet Embeddings API",
    description="API para extraer y comparar embeddings faciales usando Keras-FaceNet y PostgreSQL con pgvector",
    version="1.0.0"
)

@app.get("/health-check")
async def health_check():
    return {"status": "ok"}
    
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

def search_similar_embeddings(db: Session, embedding: List[float], threshold: float = 0.5):
    """Busca embeddings similares en la base de datos"""
    try:
        # Convertir a array PostgreSQL
        embedding_array = "[" + ",".join(map(str, embedding)) + "]"
        
        query = text("""
            SELECT v.id_vector, p.id_persona, p.nombre, p.apellido_paterno, 
                   p.activo, 1 - (v.vector <=> :embedding) as similitud
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

def register_access_attempt(db: Session, id_persona: Optional[int], confidence: Optional[float], 
                           access_granted: bool, photo_url: Optional[str] = None):
    try:
        timezone_mx = pytz.timezone('America/Mexico_City')
        now_mx = datetime.now(timezone_mx)
        
        # Convertir a string con offset explícito
        fecha_mx = now_mx.strftime('%Y-%m-%d %H:%M:%S-06:00')  # Ajusta -05:00 en horario de verano
        
        query = text("""
            INSERT INTO historial_accesos 
            (id_persona, id_dispositivo, fecha, resultado, confianza, foto_url)
            VALUES 
            (:id_persona, 3, :fecha::timestamp with time zone, :resultado, :confianza, :foto_url)
        """)
        
        db.execute(query, {
            "id_persona": id_persona,
            "fecha": fecha_mx,
            "resultado": "Éxito" if access_granted else "Fallo",
            "confianza": confidence,
            "foto_url": photo_url
        })
        db.commit()
    except Exception as e:
        logger.error(f"Error al registrar acceso: {str(e)}")
        db.rollback()
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
    - access_granted: Si se concedió el acceso
    - reason: Razón por la que se concedió o denegó el acceso
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
            # Registrar intento fallido (sin rostro detectado)
            register_access_attempt(db, None, None, False)
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")
        
        embedding = detections[0].tolist()
        
        # Buscar coincidencias en la BD
        matches = search_similar_embeddings(db, embedding)
        
        # Formatear resultados y verificar acceso
        formatted_matches = []
        best_match = None
        access_granted = False
        reason = "No se encontraron coincidencias"
        
        if matches:
            best_match = {
                "id_persona": matches[0].id_persona,
                "nombre_completo": f"{matches[0].nombre} {matches[0].apellido_paterno}",
                "similitud": float(matches[0].similitud),
                "id_vector": matches[0].id_vector,
                "activo": matches[0].activo
            }
            
            # Verificar si la persona está activa
            if matches[0].activo:
                access_granted = True
                reason = "Persona reconocida y activa"
            else:
                reason = "Persona reconocida pero inactiva"
        
        for match in matches:
            formatted_matches.append({
                "id_persona": match.id_persona,
                "nombre_completo": f"{match.nombre} {match.apellido_paterno}",
                "similitud": float(match.similitud),
                "id_vector": match.id_vector,
                "activo": match.activo
            })
        
        # Registrar el intento de acceso
        confidence = float(matches[0].similitud) if matches else None
        id_persona = matches[0].id_persona if matches else None
        register_access_attempt(db, id_persona, confidence, access_granted)
        
        return {
            "message": "Embeddings generados correctamente",
            "embedding_size": len(embedding),
            "matches": formatted_matches,
            "best_match": best_match,
            "access_granted": access_granted,
            "reason": reason
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error al procesar la imagen")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/", include_in_schema=False)
def health_check():
    return {"status": "ok", "message": "API de reconocimiento facial operativa"}
