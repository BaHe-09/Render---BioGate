from fastapi import FastAPI, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
import bcrypt
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware
from database import get_db
import json

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa la app FastAPI
app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe esto a tus dominios
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class UserLogin(BaseModel):
    username: str
    password: str

class RegistroPersona(BaseModel):
    name: str  # Nombre(s)
    lastName: str  # Primer apellido
    secondLastName: Optional[str] = None  # Segundo apellido (opcional)
    phone: str  # Teléfono completo (código + número)
    email: EmailStr  # Correo electrónico

class RegistroCuenta(BaseModel):
    password: str  # Contraseña
    confirmPassword: str  # Confirmación de contraseña

    @validator('confirmPassword')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Las contraseñas no coinciden')
        return v

class UsuarioRegistro(BaseModel):
    persona: RegistroPersona
    cuenta: RegistroCuenta

class AccessRecord(BaseModel):
    id_acceso: int
    nombre_completo: str
    fecha: datetime
    dispositivo: str
    estado_registro: str
    confianza: Optional[float]
    foto_url: Optional[str]

class User(BaseModel):
    id_persona: int
    nombre: str
    apellido_paterno: str
    correo: str
    telefono: str
    rol: str
    activo: bool

class StatsResponse(BaseModel):
    total_entries: int
    today_entries: int
    late_entries: int
    active_users: int
    recent_access: List[AccessRecord]

class Report(BaseModel):
    id_reporte: int
    titulo: str
    tipo_reporte: str
    severidad: str
    estado: str
    fecha_generacion: datetime
# --- Endpoints ---
@app.get("/")
def read_root():
    return {
        "message": "API de autenticación funcionando",
        "status": "active",
        "endpoints": {
            "login": "POST /login/",
            "register": "POST /registrar/",
            "historial": "GET /historial-accesos/",
            "generate_password": "GET /generate-password/",
            "docs": "/docs"
        }
    }

@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        logger.info(f"Intento de login para: {user.username}")

        # 1. Buscar usuario en la base de datos
        query = text("""
            SELECT id_cuenta, contrasena_hash 
            FROM cuentas 
            WHERE nombre_usuario = :username
            LIMIT 1
        """)
        result = db.execute(query, {"username": user.username})
        user_db = result.fetchone()

        if not user_db:
            logger.warning("Usuario no encontrado")
            raise HTTPException(
                status_code=401,
                detail="Credenciales inválidas",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 2. Verificar contraseña con bcrypt
        if not bcrypt.checkpw(
            user.password.encode('utf-8'),
            user_db.contrasena_hash.encode('utf-8')
        ):
            logger.warning("Contraseña incorrecta")
            raise HTTPException(
                status_code=401,
                detail="Credenciales inválidas",
                headers={"WWW-Authenticate": "Bearer"}
            )

        logger.info("Autenticación exitosa")
        return {
            "status": "success",
            "user_id": user_db.id_cuenta,
            "message": "Autenticación exitosa"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )


@app.post("/registrar/", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
    try:
        logger.info(f"Intento de registro para: {usuario.persona.email}")

        # Verificar si el correo ya existe
        correo_existente = db.execute(
            text("SELECT 1 FROM personas WHERE correo_electronico = :correo"),
            {"correo": usuario.persona.email}
        ).scalar()

        if correo_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico ya está registrado"
            )

        # Verificar si el nombre de usuario ya existe
        nombre_usuario = usuario.persona.email.split('@')[0]
        usuario_existente = db.execute(
            text("SELECT 1 FROM cuentas WHERE nombre_usuario = :username"),
            {"username": nombre_usuario}
        ).scalar()

        if usuario_existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre de usuario ya está en uso"
            )

        # Hashear contraseña
        hashed_password = bcrypt.hashpw(
            usuario.cuenta.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        # Insertar persona
        result_persona = db.execute(
            text("""
                INSERT INTO personas (
                    nombre, apellido_paterno, apellido_materno, 
                    telefono, correo_electronico, fecha_registro, activo
                ) 
                VALUES (
                    :nombre, :apellido_paterno, :apellido_materno, 
                    :telefono, :correo, :fecha_registro, TRUE
                )
                RETURNING id_persona
            """),
            {
                "nombre": usuario.persona.name,
                "apellido_paterno": usuario.persona.lastName,
                "apellido_materno": usuario.persona.secondLastName,
                "telefono": usuario.persona.phone,
                "correo": usuario.persona.email,
                "fecha_registro": datetime.now()
            }
        )
        id_persona = result_persona.scalar_one()

        # Insertar cuenta
        db.execute(
            text("""
                INSERT INTO cuentas (
                    id_persona, id_rol, nombre_usuario, 
                    contrasena_hash, sal, ultimo_acceso
                ) 
                VALUES (
                    :id_persona, 
                    1,  -- Rol de Administrador
                    :nombre_usuario, 
                    :contrasena_hash, 
                    '',  -- Sal (ya incluida en bcrypt)
                    :ultimo_acceso
                )
            """),
            {
                "id_persona": id_persona,
                "nombre_usuario": nombre_usuario,
                "contrasena_hash": hashed_password,
                "ultimo_acceso": datetime.now()
            }
        )

        db.commit()
        logger.info(f"Usuario administrador registrado exitosamente: {usuario.persona.email}")

        return {
            "status": "success",
            "id_persona": id_persona,
            "nombre_usuario": nombre_usuario,
            "message": "Usuario administrador registrado exitosamente"
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as e:
        db.rollback()
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Obtiene estadísticas generales para el dashboard"""
    try:
        # Estadísticas básicas
        total_entries = db.execute(text("SELECT COUNT(*) FROM historial_accesos")).scalar()
        today_entries = db.execute(
            text("SELECT COUNT(*) FROM historial_accesos WHERE DATE(fecha) = CURRENT_DATE")
        ).scalar()
        late_entries = db.execute(
            text("""
                SELECT COUNT(*) FROM historial_accesos 
                WHERE estado_registro LIKE 'RETRASO%'
            """)
        ).scalar()
        active_users = db.execute(
            text("SELECT COUNT(*) FROM personas WHERE activo = TRUE")
        ).scalar()

        # Últimos 10 accesos
        recent_access = db.execute(
            text("""
                SELECT h.id_acceso, 
                       CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                       h.fecha, 
                       d.nombre as dispositivo,
                       h.estado_registro,
                       h.confianza,
                       h.foto_url
                FROM historial_accesos h
                JOIN personas p ON h.id_persona = p.id_persona
                LEFT JOIN dispositivos d ON h.id_dispositivo = d.id_dispositivo
                ORDER BY h.fecha DESC
                LIMIT 10
            """)
        ).fetchall()

        return {
            "total_entries": total_entries,
            "today_entries": today_entries,
            "late_entries": late_entries,
            "active_users": active_users,
            "recent_access": recent_access
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.get("/access", response_model=List[AccessRecord])
def get_access_records(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: Optional[str] = None,
    user_id: Optional[int] = None,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Obtiene registros de acceso con filtros"""
    try:
        query = text("""
            SELECT h.id_acceso, 
                   CONCAT(p.nombre, ' ', p.apellido_paterno) as nombre_completo,
                   h.fecha, 
                   d.nombre as dispositivo,
                   h.estado_registro,
                   h.confianza,
                   h.foto_url
            FROM historial_accesos h
            JOIN personas p ON h.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON h.id_dispositivo = d.id_dispositivo
            WHERE 1=1
        """)

        params = {}
        conditions = []

        if start_date:
            conditions.append("h.fecha >= :start_date")
            params["start_date"] = start_date
        if end_date:
            conditions.append("h.fecha <= :end_date")
            params["end_date"] = end_date
        if status:
            conditions.append("h.estado_registro = :status")
            params["status"] = status
        if user_id:
            conditions.append("h.id_persona = :user_id")
            params["user_id"] = user_id

        if conditions:
            query = text(f"{query.text} {' AND '.join(conditions)}")

        query = text(f"{query.text} ORDER BY h.fecha DESC LIMIT :limit OFFSET :offset")
        params["limit"] = limit
        params["offset"] = (page - 1) * limit

        results = db.execute(query, params).fetchall()
        return results

    except Exception as e:
        logger.error(f"Error getting access records: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving access records")

@app.get("/users", response_model=List[User])
def get_users(
    active_only: bool = True,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Obtiene lista de usuarios"""
    try:
        query = text("""
            SELECT p.id_persona, p.nombre, p.apellido_paterno, 
                   p.correo_electronico as correo, p.telefono,
                   r.nombre as rol, p.activo
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            JOIN roles r ON c.id_rol = r.id_rol
            WHERE (:active_only = FALSE OR p.activo = TRUE)
        """)

        params = {"active_only": active_only}

        if search:
            query = text(f"{query.text} AND (p.nombre ILIKE :search OR p.apellido_paterno ILIKE :search OR p.correo_electronico ILIKE :search)")
            params["search"] = f"%{search}%"

        results = db.execute(query, params).fetchall()
        return results

    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving users")

@app.get("/reports", response_model=List[Report])
def get_reports(
    report_type: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Obtiene reportes filtrados"""
    try:
        query = text("""
            SELECT id_reporte, titulo, tipo_reporte, severidad, estado, fecha_generacion
            FROM reportes
            WHERE 1=1
        """)

        params = {}
        conditions = []

        if report_type:
            conditions.append("tipo_reporte = :report_type")
            params["report_type"] = report_type
        if status:
            conditions.append("estado = :status")
            params["status"] = status
        if severity:
            conditions.append("severidad = :severity")
            params["severity"] = severity

        if conditions:
            query = text(f"{query.text} {' AND '.join(conditions)}")

        query = text(f"{query.text} ORDER BY fecha_generacion DESC LIMIT 50")

        results = db.execute(query, params).fetchall()
        return results

    except Exception as e:
        logger.error(f"Error getting reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving reports")

@app.get("/reports/stats")
def get_report_stats(db: Session = Depends(get_db)):
    """Obtiene estadísticas de reportes"""
    try:
        today = datetime.now().date()
        
        # Reportes de hoy
        today_reports = db.execute(
            text("SELECT COUNT(*) FROM reportes WHERE DATE(fecha_generacion) = :today"),
            {"today": today}
        ).scalar()
        
        # Accesos no autorizados
        unauthorized_access = db.execute(
            text("""
                SELECT COUNT(*) FROM reportes 
                WHERE tipo_reporte = 'Acceso no autorizado' 
                AND estado != 'Cerrado'
            """)
        ).scalar()
        
        # Reportes abiertos
        open_reports = db.execute(
            text("SELECT COUNT(*) FROM reportes WHERE estado IN ('Abierto', 'En progreso')")
        ).scalar()

        return {
            "today_reports": today_reports,
            "unauthorized_access": unauthorized_access,
            "open_reports": open_reports
        }

    except Exception as e:
        logger.error(f"Error getting report stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving report statistics")

@app.put("/users/{user_id}/status")
def update_user_status(
    user_id: int,
    active: bool,
    db: Session = Depends(get_db)
):
    """Activa/desactiva un usuario"""
    try:
        db.execute(
            text("UPDATE personas SET activo = :active WHERE id_persona = :user_id"),
            {"active": active, "user_id": user_id}
        )
        db.commit()
        return {"message": "User status updated successfully"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user status")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
