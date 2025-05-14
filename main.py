import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from facenet_utils import load_facenet_model, get_embedding, compare_with_db
from typing import List, Optional
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(
    title="Face Recognition API",
    description="API para comparar rostros con embeddings en Neontech DB",
    version="1.0.0"
)

# Cargar modelo Facenet al iniciar (se cachea automáticamente)
facenet_model = load_facenet_model()

@app.on_event("startup")
async def startup_event():
    # Validar que existe la variable de conexión
    if not os.getenv("NEON_DB_URI"):
        raise RuntimeError("NEON_DB_URI no está configurado en las variables de entorno")

@app.get("/")
def health_check():
    return {
        "status": "OK",
        "message": "Face Recognition API with Facenet",
        "python_version": "3.9.16",
        "endpoints": {
            "/compare-face": "POST - Compara un rostro con la base de datos",
            "/docs": "Interfaz Swagger UI"
        }
    }

@app.post("/compare-face")
async def compare_face(
    file: UploadFile = File(..., description="Imagen con rostro a comparar (JPEG/PNG)"),
    threshold: float = 0.7,
    top_k: int = 5
):
    """
    Compara un rostro con la base de datos de vectores faciales.
    
    Parámetros:
    - threshold: Umbral de similitud (0.0 a 1.0)
    - top_k: Número máximo de coincidencias a devolver
    
    Retorna:
    - Lista de coincidencias ordenadas por similitud descendente
    """
    try:
        # Validar tipo de archivo
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Formato de imagen no soportado. Use JPEG o PNG")

        # Leer y procesar imagen
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        # Obtener embedding con Facenet
        embedding = get_embedding(facenet_model, img)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro en la imagen")
        
        # Comparar con la base de datos
        matches = compare_with_db(embedding, threshold, top_k)
        
        return JSONResponse(content={
            "status": "success",
            "matches": matches,
            "embedding_size": len(embedding),
            "threshold": threshold,
            "top_k_requested": top_k,
            "matches_returned": len(matches)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10001))  # Puerto diferente al de la otra API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        reload=False  # Desactivar en producción
    )
