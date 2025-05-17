from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from app.model_utils import detect_face, get_embedding, recognize

app = FastAPI()

@app.post("/identificar/")
async def identificar(file: UploadFile = File(...)):
    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    face = detect_face(image)
    if face is None:
        return {"status": "error", "detail": "No se detectó rostro"}
    
    emb = get_embedding(face)
    id_persona, confianza = recognize(emb)
    
    return {
        "status": "ok",
        "id_persona": id_persona,
        "confianza": confianza
    }
