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
from datetime import datetime, time, timedelta
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

def obtener_horario_persona(db: Session, id_persona: int):
    """Obtiene el horario laboral de una persona desde la BD"""
    try:
        query = text("""
            SELECT hora_entrada, hora_salida, tolerancia_retraso
            FROM horarios_persona
            WHERE id_persona = :id_persona
        """)
        result = db.execute(query, {"id_persona": id_persona})
        horario = result.fetchone()
        
        if not horario:
            # Horario por defecto si no está configurado
            return time(8, 0), time(17, 0), 10  # 8:00 AM a 5:00 PM, 10 mins tolerancia
        
        return horario.hora_entrada, horario.hora_salida, horario.tolerancia_retraso
        
    except Exception as e:
        logger.error(f"Error al obtener horario: {str(e)}")
        return time(8, 0), time(17, 0), 10  # Valores por defecto en caso de error

def determinar_estado_registro(hora_registro: time, hora_entrada: time, 
                              hora_salida: time, tolerancia: int) -> str:
    """
    Determina el estado de registro basado en los horarios de la persona.
    
    Reglas:
    - Si es antes de hora_entrada + tolerancia: "ENTRADA"
    - Si es después de hora_entrada + tolerancia pero en horario laboral: "RETRASO"
    - Si es después de hora_salida: "SALIDA"
    - Si es fuera del horario laboral: "HORAS_EXTRAS"
    """
    # Calcular hora límite para entrada normal
    hora_limite_entrada = (
        datetime.combine(datetime.today(), hora_entrada) + 
        timedelta(minutes=tolerancia)
    ).time()
    
    if hora_registro <= hora_limite_entrada:
        return "ENTRADA"
    elif hora_registro <= hora_salida:
        return "RETRASO"
    else:
        return "SALIDA"

def register_access_attempt(db: Session, id_persona: Optional[int], confidence: Optional[float], 
                           access_granted: bool, photo_url: Optional[str] = None):
    """Registra un intento de acceso en el historial"""
    try:
        timezone_mx = pytz.timezone('America/Mexico_City')
        now_mx = datetime.now(timezone_mx)
        hora_actual = now_mx.time()
        
        # Determinar el estado del registro
        estado_registro = "DESCONOCIDO"
        if id_persona and access_granted:
            hora_entrada, hora_salida, tolerancia = obtener_horario_persona(db, id_persona)
            estado_registro = determinar_estado_registro(hora_actual, hora_entrada, hora_salida, tolerancia)
        
        query = text("""
            INSERT INTO historial_accesos 
            (id_persona, id_dispositivo, fecha, resultado, confianza, foto_url, estado_registro)
            VALUES 
            (:id_persona, 3, :fecha, :resultado, :confianza, :foto_url, :estado_registro)
        """)
        
        db.execute(query, {
            "id_persona": id_persona,
            "fecha": now_mx,
            "resultado": "Éxito" if access_granted else "Fallo",
            "confianza": confidence,
            "foto_url": photo_url,
            "estado_registro": estado_registro
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
    
    Retorna:
    - estado_registro: ENTRADA, SALIDA, RETRASO o HORAS_EXTRAS según el horario de la persona
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
            register_access_attempt(db, None, None, False)
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")
        
        embedding = detections[0].tolist()
        matches = search_similar_embeddings(db, embedding)
        
        # Procesar coincidencias
        formatted_matches = []
        best_match = None
        access_granted = False
        reason = "No se encontraron coincidencias"
        estado_registro = None
        horario_info = None
        
        if matches:
            best_match = {
                "id_persona": matches[0].id_persona,
                "nombre_completo": f"{matches[0].nombre} {matches[0].apellido_paterno}",
                "similitud": float(matches[0].similitud),
                "activo": matches[0].activo
            }
            
            if matches[0].activo:
                access_granted = True
                reason = "Persona reconocida y activa"
                
                # Obtener horario y determinar estado
                hora_entrada, hora_salida, tolerancia = obtener_horario_persona(db, matches[0].id_persona)
                timezone_mx = pytz.timezone('America/Mexico_City')
                hora_actual = datetime.now(timezone_mx).time()
                estado_registro = determinar_estado_registro(
                    hora_actual, hora_entrada, hora_salida, tolerancia)
                
                horario_info = {
                    "hora_entrada": hora_entrada.strftime("%H:%M"),
                    "hora_salida": hora_salida.strftime("%H:%M"),
                    "tolerancia_retraso": tolerancia,
                    "hora_registro": hora_actual.strftime("%H:%M")
                }
            else:
                reason = "Persona reconocida pero inactiva"
        
        # Registrar el intento de acceso
        confidence = float(matches[0].similitud) if matches else None
        id_persona = matches[0].id_persona if matches else None
        register_access_attempt(db, id_persona, confidence, access_granted)
        
        response_data = {
            "message": "Embeddings generados correctamente",
            "access_granted": access_granted,
            "reason": reason,
            "estado_registro": estado_registro,
            "hora_registro": datetime.now().strftime("%H:%M:%S")
        }
        
        if best_match:
            response_data["best_match"] = best_match
        if horario_info:
            response_data["horario_info"] = horario_info
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error al procesar la imagen")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
