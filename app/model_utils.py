import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from app.db import get_vectors, log_access

yolo = YOLO("models/yolov8n-face-lindevs.pt")
facenet = tf.keras.models.load_model("models/facenet_model.h5")

def detect_face(image_np):
    results = yolo(image_np)[0]
    if results.boxes:
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        return image_np[box[1]:box[3], box[0]:box[2]]
    return None

def get_embedding(face_img):
    resized = cv2.resize(face_img, (160, 160))
    norm = resized / 255.0
    embedding = facenet.predict(np.expand_dims(norm, axis=0))[0]
    return embedding

def recognize(embedding, threshold=0.6):
    vectors = get_vectors()
    ids, stored_vectors = zip(*vectors)
    sims = cosine_similarity([embedding], stored_vectors)[0]
    max_idx = np.argmax(sims)
    if sims[max_idx] > threshold:
        log_access(ids[max_idx], float(sims[max_idx]), "Éxito")
        return ids[max_idx], float(sims[max_idx])
    else:
        log_access(None, float(sims[max_idx]), "Desconocido")
        return None, float(sims[max_idx])
