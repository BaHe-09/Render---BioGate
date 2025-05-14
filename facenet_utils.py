import os
from urllib.parse import urlparse
from psycopg2 import pool  # Añadir esto al inicio del archivo

# Conexión pool para mejor manejo de conexiones
connection_pool = None

def init_db_pool():
    global connection_pool
    db_uri = os.getenv("NEON_DB_URI")
    if not db_uri:
        raise ValueError("NEON_DB_URI no está configurado en las variables de entorno")
    
    # Parsear la URI
    result = urlparse(db_uri)
    username = result.username
    password = result.password
    database = result.path[1:]  # Elimina el '/' inicial
    hostname = result.hostname
    port = result.port or 5432
    
    connection_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        user=username,
        password=password,
        host=hostname,
        port=port,
        database=database,
        sslmode="require"  # NeonTech requiere SSL
    )

def compare_with_db(embedding, threshold=0.7, top_k=5):
    global connection_pool
    
    if connection_pool is None:
        init_db_pool()
    
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
        conn = connection_pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (embedding, embedding, threshold, top_k))
            results = cur.fetchall()
            return results
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            connection_pool.putconn(conn)
