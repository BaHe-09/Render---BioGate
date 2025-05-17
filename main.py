from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from facenet_keras import models
from tensorflow.keras.preprocessing import image

app = FastAPI(title="FaceNet Embeddings API")

# Cargar el modelo FaceNet al iniciar la aplicación
facenet_model = models.load_model()

def preprocess_image(img):
    """Preprocesa la imagen para el modelo FaceNet"""
    img = img.resize((160, 160))  # FaceNet espera imágenes de 160x160
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalización según lo esperado por FaceNet
    mean = np.mean(img_array)
    std = np.std(img_array)
    std_adj = np.maximum(std, 1.0/np.sqrt(img_array.size))
    img_array = (img_array - mean) / std_adj
    return img_array

@app.post("/get_embeddings")
async def get_face_embeddings(file: UploadFile = File(...)):
    # Verificar que el archivo es una imagen
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Leer la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Preprocesar la imagen
        processed_img = preprocess_image(img)
        
        # Obtener los embeddings
        embeddings = facenet_model.predict(processed_img)
        
        # Convertir a lista para la respuesta JSON
        embeddings_list = embeddings[0].tolist()
        
        return JSONResponse(content={"embeddings": embeddings_list})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "FaceNet Embeddings API - Sube una imagen para obtener embeddings faciales"}
