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
from datetime import datetime, timezone, time
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
                           access_granted: bool, photo_url: Optional[str] = None, 
                           tipo_registro: str = "Entrada"):
    """Registra un intento de acceso en el historial y actualiza asistencia"""
    try:
        timezone_mx = pytz.timezone('America/Mexico_City')
        now_mx = datetime.now(timezone_mx)
        fecha_mx = now_mx.isoformat()
        
        # Insertar en historial_accesos
        query = text("""
            INSERT INTO historial_accesos 
            (id_persona, id_dispositivo, fecha, resultado, confianza, foto_url, tipo_registro)
            VALUES 
            (:id_persona, 3, :fecha, :resultado, :confianza, :foto_url, :tipo_registro)
            RETURNING id_acceso
        """)
        
        result = db.execute(query, {
            "id_persona": id_persona,
            "fecha": fecha_mx,
            "resultado": "Éxito" if access_granted else "Fallo",
            "confianza": confidence,
            "foto_url": photo_url,
            "tipo_registro": tipo_registro
        })
        
        id_acceso = result.scalar()
        db.commit()
        
        return id_acceso
        
    except Exception as e:
        logger.error(f"Error al registrar acceso: {str(e)}")
        db.rollback()
        raise

@app.post("/get_embeddings", response_model=dict)
async def get_face_embeddings(
    file: UploadFile = File(...),
    tipo_registro: str = "Entrada",
    db: Session = Depends(get_db)
):
    """
    Endpoint para reconocimiento facial y control de asistencia
    
    Parámetros:
    - file: Archivo de imagen con el rostro
    - tipo_registro: 'Entrada' (default) o 'Salida' para control de asistencia
    
    Retorna:
    - access_granted: Booleano indicando si se concedió acceso
    - persona: Datos de la persona reconocida (si aplica)
    - estado_asistencia: Puntual/Retraso/Falta (solo para entradas)
    - detalles_turno: Información del turno asignado
    - metadata: Información técnica del reconocimiento
    """
    # Validación inicial
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    if tipo_registro not in ["Entrada", "Salida"]:
        raise HTTPException(status_code=400, detail="tipo_registro debe ser 'Entrada' o 'Salida'")

    try:
        # 1. Procesamiento de imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_array = preprocess_image(img)
        
        # 2. Extracción de embeddings faciales
        detections = facenet.embeddings([img_array])
        if len(detections) == 0:
            register_access_attempt(db, None, None, False, None, tipo_registro)
            raise HTTPException(status_code=400, detail="No se detectaron rostros en la imagen")
        
        embedding = detections[0].tolist()
        
        # 3. Búsqueda en base de datos
        matches = search_similar_embeddings(db, embedding, threshold=0.5)
        if not matches:
            register_access_attempt(db, None, None, False, None, tipo_registro)
            return {
                "access_granted": False,
                "reason": "No se encontraron coincidencias",
                "metadata": {
                    "embedding_size": len(embedding),
                    "confidence": None
                }
            }

        best_match = matches[0]
        confidence = float(best_match.similitud)
        
        # 4. Verificación de acceso
        access_granted = best_match.activo
        reason = "Persona reconocida y activa" if access_granted else "Persona reconocida pero inactiva"
        
        # 5. Registro del intento de acceso
        id_acceso = register_access_attempt(
            db, 
            best_match.id_persona, 
            confidence, 
            access_granted, 
            None, 
            tipo_registro
        )
        
        # 6. Información de asistencia (solo si acceso concedido)
        estado_asistencia = None
        detalles_turno = None
        
        if access_granted:
            # Obtener información del turno y estado de asistencia
            query = text("""
                SELECT 
                    t.nombre AS nombre_turno,
                    t.hora_entrada,
                    t.hora_salida,
                    determinar_estado_asistencia(:id_persona, NOW()) AS estado,
                    CURRENT_TIME < t.hora_entrada AS es_antes_turno,
                    CURRENT_TIME > t.hora_salida AS es_despues_turno
                FROM persona_turnos pt
                JOIN turnos t ON pt.id_turno = t.id_turno
                WHERE pt.id_persona = :id_persona
                AND (pt.fecha_fin IS NULL OR pt.fecha_fin >= CURRENT_DATE)
                AND pt.fecha_inicio <= CURRENT_DATE
                AND (
                    SELECT EXTRACT(DOW FROM CURRENT_TIMESTAMP)::INT = ANY(
                        string_to_array(pt.dias_semana, ',')::int[]
                    )
                LIMIT 1
            """)
            
            result = db.execute(query, {"id_persona": best_match.id_persona})
            turno_info = result.fetchone()
            
            if turno_info:
                estado_asistencia = turno_info.estado
                detalles_turno = {
                    "nombre": turno_info.nombre_turno,
                    "hora_entrada": str(turno_info.hora_entrada),
                    "hora_salida": str(turno_info.hora_salida),
                    "fuera_de_horario": (
                        turno_info.es_antes_turno if tipo_registro == "Entrada" 
                        else turno_info.es_despues_turno
                    )
                }

        # 7. Preparar respuesta
        response_data = {
            "access_granted": access_granted,
            "persona": {
                "id": best_match.id_persona,
                "nombre_completo": f"{best_match.nombre} {best_match.apellido_paterno}",
                "activo": best_match.activo
            },
            "estado_asistencia": estado_asistencia,
            "detalles_turno": detalles_turno,
            "metadata": {
                "confidence": confidence,
                "embedding_size": len(embedding),
                "match_count": len(matches),
                "tipo_registro": tipo_registro,
                "timestamp": datetime.now(pytz.timezone('America/Mexico_City')).isoformat()
            }
        }
        
        # 8. Log para depuración
        logger.info(
            f"Acceso {'APROBADO' if access_granted else 'DENEGADO'} | "
            f"Persona: {best_match.id_persona} | "
            f"Confianza: {confidence:.2f} | "
            f"Tipo: {tipo_registro} | "
            f"Estado: {estado_asistencia or 'N/A'}"
        )
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error en el procesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

def verificar_horario_turno(db: Session, id_persona: int):
    """Verifica si el acceso está dentro del horario del turno asignado"""
    query = text("""
        SELECT 
            t.hora_entrada,
            t.hora_salida,
            CURRENT_TIME BETWEEN t.hora_entrada AND t.hora_salida AS dentro_horario
        FROM persona_turnos pt
        JOIN turnos t ON pt.id_turno = t.id_turno
        WHERE pt.id_persona = :id_persona
        AND (pt.fecha_fin IS NULL OR pt.fecha_fin >= CURRENT_DATE)
        AND pt.fecha_inicio <= CURRENT_DATE
        AND (SELECT EXTRACT(DOW FROM CURRENT_TIMESTAMP)::INT = ANY(
            SELECT unnest(string_to_array(pt.dias_semana, ',')::INT[])
        )
        LIMIT 1
    """)
    
    result = db.execute(query, {"id_persona": id_persona})
    return result.fetchone()
    
@app.get("/", include_in_schema=False)
def health_check():
    return {"status": "ok", "message": "API de reconocimiento facial operativa"}
