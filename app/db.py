from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.config import DB_URL

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def get_vectors():
    with SessionLocal() as session:
        result = session.execute(text("SELECT id_persona, vector FROM vectores_identificacion"))
        return [(r.id_persona, r.vector) for r in result]

def log_access(id_persona, confianza, resultado):
    with SessionLocal() as session:
        session.execute(text("""
            INSERT INTO historial_accesos (id_persona, resultado, confianza)
            VALUES (:id, :res, :conf)
        """), {"id": id_persona, "res": resultado, "conf": confianza})
        session.commit()
