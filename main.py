from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from keras_facenet import FaceNet
from typing import List

app = FastAPI(
    title="FaceNet Embeddings API",
    description="API para extraer embeddings faciales usando Keras-FaceNet",
    version="1.0.0"
)

# Cargar el modelo FaceNet (se descarga automáticamente la primera vez)
facenet = FaceNet()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convierte la imagen al formato requerido por FaceNet"""
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convertir a array numpy
    img_array = np.array(img)
    return img_array

@app.post("/get_embeddings", response_model=dict)
async def get_face_embeddings(file: UploadFile = File(...)):
    """
    Endpoint para obtener embeddings faciales de una imagen
    
    Parámetros:
    - file: Archivo de imagen que contiene un rostro
    
    Retorna:
    - embeddings: Vector de 512 dimensiones con las características faciales
    """
    # Validar que sea una imagen
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="El archivo debe ser una imagen (JPEG, PNG, etc.)"
        )
    
    try:
        # Leer la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocesar imagen
        img_array = preprocess_image(img)
        
        # Obtener embeddings (el modelo detecta rostros automáticamente)
        detections = facenet.embeddings([img_array])
        
        if len(detections) == 0:
            raise HTTPException(
                status_code=400,
                detail="No se detectaron rostros en la imagen"
            )
        
        # Tomar el primer rostro detectado
        embeddings = detections[0].tolist()
        
        return {
            "message": "Embeddings generados correctamente",
            "embeddings": embeddings,
            "embedding_size": len(embeddings)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

@app.get("/", include_in_schema=False)
def health_check():
    return {"status": "ok", "message": "API de reconocimiento facial operativa"}
