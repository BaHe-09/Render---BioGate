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

def obtener_horario_persona(db: Session, id_persona: int, fecha_actual: datetime):
    """Obtiene el horario laboral de una persona desde la BD considerando el día de la semana"""
    try:
        # Primero obtener el día de la semana (0=lunes, 6=domingo)
        dia_semana = fecha_actual.weekday()
        
        query = text("""
            SELECT hora_entrada, hora_salida, tolerancia_retraso, dias_laborales
            FROM horarios_persona
            WHERE id_persona = :id_persona
        """)
        result = db.execute(query, {"id_persona": id_persona})
        horario = result.fetchone()
        
        if not horario:
            # Horario por defecto si no está configurado (L-V de 8am a 5pm)
            return time(8, 0), time(17, 0), 10, 'L-V'
        
        # Verificar si es día laboral
        dias_validos = {
            'L-V': [0, 1, 2, 3, 4],    # Lunes a Viernes
            'L-S': [0, 1, 2, 3, 4, 5],  # Lunes a Sábado
            'L-D': [0, 1, 2, 3, 4, 5, 6] # Lunes a Domingo
        }
        
        dias_permitidos = dias_validos.get(horario.dias_laborales, [0, 1, 2, 3, 4])
        
        if dia_semana not in dias_permitidos:
            # Si no es día laboral, considerar como horas extras
            return None, None, 0, horario.dias_laborales
        
        return horario.hora_entrada, horario.hora_salida, horario.tolerancia_retraso, horario.dias_laborales
        
    except Exception as e:
        logger.error(f"Error al obtener horario: {str(e)}")
        return time(8, 0), time(17, 0), 10, 'L-V'  # Valores por defecto en caso de error

def determinar_estado_registro(hora_registro: time, hora_entrada: time, 
                             hora_salida: time, tolerancia: int, 
                             dias_laborales: str, dia_actual: int) -> str:
    """
    Determina el estado de registro considerando días laborales
    
    Args:
        dia_actual: 0=lunes, 6=domingo
    """
    # Si no hay horario (día no laboral)
    if hora_entrada is None or hora_salida is None:
        return "HORAS_EXTRAS"
    
    # Convertir a datetime para cálculos
    registro_dt = datetime.combine(datetime.today(), hora_registro)
    entrada_dt = datetime.combine(datetime.today(), hora_entrada)
    salida_dt = datetime.combine(datetime.today(), hora_salida)
    
    # Calcular límites
    hora_limite_entrada = entrada_dt + timedelta(minutes=tolerancia)
    hora_limite_salida = salida_dt + timedelta(hours=2)
    
    if registro_dt < entrada_dt:
        return "HORAS_EXTRAS"
    elif registro_dt <= hora_limite_entrada:
        return "ENTRADA"
    elif registro_dt <= salida_dt:
        return "RETRASO"
    elif registro_dt <= hora_limite_salida:
        return "SALIDA"
    else:
        return "HORAS_EXTRAS"

def register_access_attempt(db: Session, id_persona: Optional[int], confidence: Optional[float], 
                         access_granted: bool, photo_url: Optional[str] = None):
    """Registra un intento de acceso en el historial"""
    try:
        timezone_mx = pytz.timezone('America/Mexico_City')
        now_mx = datetime.now(timezone_mx)
        hora_actual_mx = now_mx.time()
        dia_actual = now_mx.weekday()  # 0=lunes, 6=domingo
        
        estado_registro = "DESCONOCIDO"
        horas_extras = 0
        es_dia_laboral = True

        if id_persona and access_granted:
            hora_entrada, hora_salida, tolerancia, dias_laborales = obtener_horario_persona(db, id_persona, now_mx)
            
            if hora_entrada is None:  # No es día laboral
                estado_registro = "HORAS_EXTRAS"
                es_dia_laboral = False
                # Calcular horas extras completas (jornada completa fuera de día laboral)
                horas_extras = 8.0  # 8 horas extras por día no laboral
            else:
                estado_registro = determinar_estado_registro(
                    hora_actual_mx, hora_entrada, hora_salida, 
                    tolerancia, dias_laborales, dia_actual)
                
                if estado_registro == "HORAS_EXTRAS":
                    registro_dt = datetime.combine(datetime.today(), hora_actual_mx)
                    entrada_dt = datetime.combine(datetime.today(), hora_entrada)
                    salida_dt = datetime.combine(datetime.today(), hora_salida)
                    
                    if registro_dt < entrada_dt:
                        horas_extras = (entrada_dt - registro_dt).total_seconds() / 3600
                    else:
                        horas_extras = (registro_dt - (salida_dt + timedelta(hours=2))).total_seconds() / 3600
        
        query = text("""
            INSERT INTO historial_accesos 
            (id_persona, id_dispositivo, fecha, resultado, confianza, 
             foto_url, estado_registro, horas_extras, es_dia_laboral)
            VALUES 
            (:id_persona, 3, :fecha, :resultado, :confianza, 
             :foto_url, :estado_registro, :horas_extras, :es_dia_laboral)
        """)
        
        db.execute(query, {
            "id_persona": id_persona,
            "fecha": now_mx,
            "resultado": "Éxito" if access_granted else "Fallo",
            "confianza": confidence,
            "foto_url": photo_url,
            "estado_registro": estado_registro,
            "horas_extras": round(horas_extras, 2),
            "es_dia_laboral": es_dia_laboral
        })
        db.commit()
        
        return hora_actual_mx.strftime("%H:%M:%S")
    except Exception as e:
        logger.error(f"Error al registrar acceso: {str(e)}")
        db.rollback()
        raise

@app.post("/get_embeddings", response_model=dict)
async def get_face_embeddings(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        hora_registro_str = None
        
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
                
                # Obtener hora actual en México
                timezone_mx = pytz.timezone('America/Mexico_City')
                now_mx = datetime.now(timezone_mx)
                hora_registro_str = now_mx.strftime("%H:%M:%S")
                
                # Obtener horario y determinar estado
                hora_entrada, hora_salida, tolerancia, dias_laborales = obtener_horario_persona(db, matches[0].id_persona, now_mx)
                estado_registro = determinar_estado_registro(
                    now_mx.time(), hora_entrada, hora_salida, 
                    tolerancia, dias_laborales, now_mx.weekday())
                
                horario_info = {
                    "hora_entrada": hora_entrada.strftime("%H:%M") if hora_entrada else "N/A",
                    "hora_salida": hora_salida.strftime("%H:%M") if hora_salida else "N/A",
                    "tolerancia_retraso": tolerancia,
                    "dias_laborales": dias_laborales,
                    "hora_registro": now_mx.strftime("%H:%M")  # Hora en formato corto
                }
                
                # Registrar el acceso (devuelve la hora usada)
                hora_registro_used = register_access_attempt(
                    db, matches[0].id_persona, 
                    float(matches[0].similitud), 
                    access_granted)
                
                # Asegurarnos de usar la misma hora en toda la respuesta
                hora_registro_str = hora_registro_used or hora_registro_str
            else:
                reason = "Persona reconocida pero inactiva"
        
        response_data = {
            "message": "Embeddings generados correctamente",
            "access_granted": access_granted,
            "reason": reason,
            "estado_registro": estado_registro,
            "hora_registro": hora_registro_str,
            "dia_semana": now_mx.strftime("%A") if matches else None
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
