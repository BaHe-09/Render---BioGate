import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Cargar modelos una vez
def load_facenet_model():
    # Modelo Facenet (asegúrate de tener el modelo .h5 en tu proyecto)
    facenet_model = load_model('facenet_keras.h5', compile=False)
    return facenet_model

# Preprocesamiento de imagen para Facenet
def preprocess_face(face, required_size=(160, 160)):
    face = cv2.resize(face, required_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

# Obtener embedding de una imagen
def get_embedding(model, image):
    # Detectar rostros con MTCNN
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if not faces:
        return None
    
    # Tomar el rostro principal (el de mayor confianza)
    main_face = max(faces, key=lambda x: x['confidence'])
    x, y, w, h = main_face['box']
    face = image[y:y+h, x:x+w]
    
    # Preprocesar y obtener embedding
    face = preprocess_face(face)
    embedding = model.predict(face)[0]
    return embedding.tolist()

# Comparar con vectores en la base de datos
def compare_with_db(db_config, embedding, threshold=0.7, top_k=5):
    query = """
    SELECT 
        v.id_vector,
        p.id_persona,
        p.nombre,
        p.apellido_paterno,
        p.apellido_materno,
        1 - (v.vector <=> %s) as similarity
    FROM vectores_identificacion v
    JOIN personas p ON v.id_persona = p.id_persona
    WHERE 1 - (v.vector <=> %s) > %s
    ORDER BY similarity DESC
    LIMIT %s
    """
    
    try:
        with psycopg2.connect(**db_config, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (embedding, embedding, threshold, top_k))
                results = cur.fetchall()
                return results
    except Exception as e:
        print(f"Database error: {e}")
        return []
