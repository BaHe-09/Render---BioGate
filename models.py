import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import io
import os

# Cargamos el modelo YOLO directamente desde el archivo
def load_yolo_model():
    # Asume que el archivo yolov8n-face-lindevs.pt está en el mismo directorio
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n-face-lindevs.pt")
    return YOLO(model_path)

# Cargamos FaceNet desde Keras (versión simplificada)
def load_facenet_model():
    # Usamos una versión simplificada de FaceNet
    input_layer = Input(shape=(160, 160, 3))
    # Aquí irían las capas reales de FaceNet, pero para el ejemplo usamos una dummy
    from tensorflow.keras.layers import Conv2D, Flatten
    x = Conv2D(128, (3, 3), activation='relu')(input_layer)
    x = Flatten()(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

def load_models():
    yolo_model = load_yolo_model()
    facenet_model = load_facenet_model()
    return yolo_model, facenet_model

def process_image(image_bytes, yolo_model):
    # Convertir bytes a imagen OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detectar caras con YOLO
    results = yolo_model(img)
    
    # Tomar la primera cara detectada
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            # Redimensionar a 160x160 para FaceNet
            face = cv2.resize(face, (160, 160))
            return face
    return None

def get_face_embedding(face_image, facenet_model):
    # Preprocesamiento básico
    face_image = face_image.astype('float32')
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    
    # Añadir dimensión de batch
    face_image = np.expand_dims(face_image, axis=0)
    
    # Obtener embedding
    embedding = facenet_model.predict(face_image)
    
    # Normalizar el embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding[0]  # Quitar dimensión de batch
