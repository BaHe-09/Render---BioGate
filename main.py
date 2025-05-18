from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps
import io
from keras_facenet import FaceNet
from typing import List
import logging
import os

# Configuración inicial
app = FastAPI(
    title="FaceNet Embeddings API",
    description="API para reconocimiento facial con optimización de memoria",
    version="2.0"
)

# Configuración
facenet = FaceNet()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Límites configurables desde variables de entorno
MAX_IMAGE_SIZE_KB = int(os.getenv("MAX_IMAGE_SIZE_KB", 2048))  # 2MB por defecto
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 1024 * 1024))  # 1MP por defecto

def optimize_image(image: Image.Image) -> Image.Image:
    """Redimensiona y optimiza la imagen para FaceNet"""
    # Redimensionar si es demasiado grande
    if image.width * image.height > MAX_PIXELS:
        ratio = (MAX_PIXELS / (image.width * image.height)) ** 0.5
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Imagen redimensionada a {new_size}")
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

@app.post("/get_embeddings", response_model=dict)
async def get_face_embeddings(file: UploadFile = File(...)):
    """
    Endpoint optimizado para manejar imágenes grandes eficientemente
    """
    try:
        # 1. Validar tamaño del archivo
        contents = await file.read()
        if len(contents) > MAX_IMAGE_SIZE_KB * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Tamaño de imagen excede el límite de {MAX_IMAGE_SIZE_KB}KB"
            )
        
        # 2. Procesar imagen
        try:
            img = Image.open(io.BytesIO(contents))
            img = optimize_image(img)
            img_array = np.array(img)
        except Exception as e:
            logger.error(f"Error procesando imagen: {str(e)}")
            raise HTTPException(400, "Formato de imagen no válido")
        
        # 3. Generar embeddings
        try:
            detections = facenet.embeddings([img_array])
            if len(detections) == 0:
                raise HTTPException(400, "No se detectaron rostros")
            
            embeddings = detections[0].tolist()
            
            # Simulación de búsqueda en BD
            matches = [{
                "id_persona": 10,
                "nombre_completo": "Usuario Ejemplo",
                "similitud": 0.85,
                "id_vector": 1
            }]
            
            return {
                "message": "Embeddings generados correctamente",
                "embedding_size": len(embeddings),
                "matches": matches,
                "original_size": f"{img.width}x{img.height}",
                "processed_size": "160x160"  # FaceNet trabaja con este tamaño
            }
            
        except Exception as e:
            logger.error(f"Error en FaceNet: {str(e)}")
            raise HTTPException(500, "Error en el modelo de reconocimiento")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error inesperado")
        raise HTTPException(500, f"Error interno: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "memory_usage": f"{MAX_IMAGE_SIZE_KB}KB max"}

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API de reconocimiento facial operativa"}
