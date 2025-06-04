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

class ResumenAccesos(BaseModel):
    total_entradas: int
    total_retrasos: int
    total_salidas: int
    total_reportes: int
    usuarios_activos: int
    ultima_actualizacion: datetime

class FiltroHistorial(BaseModel):
    nombre_usuario: Optional[str] = None
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None
    estado_registro: Optional[str] = None

class UsuarioCompleto(BaseModel):
    id_persona: int
    nombre: str
    apellido_paterno: str
    apellido_materno: Optional[str]
    telefono: str
    correo_electronico: str
    activo: bool
    nombre_usuario: str
    rol: str
    hora_entrada: Optional[str]
    hora_salida: Optional[str]
    accesos: List[dict]
    total_accesos: int

class FiltroReportes(BaseModel):
    estado: Optional[str] = None
    tipo_reporte: Optional[str] = None
    severidad: Optional[str] = None
    limit: int = Field(5, ge=1, le=10) 

class FiltroUsuario(BaseModel):
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    limit: int = Field(10, ge=1, le=10)  
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

@app.get("/resumen/", response_model=ResumenAccesos)
def obtener_resumen(db: Session = Depends(get_db)):
    """Endpoint para obtener un resumen general del sistema"""
    try:
        # Obtener total de entradas
        query_entradas = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'ENTRADA%'
        """)
        total_entradas = db.execute(query_entradas).scalar_one()

        # Obtener total de retrasos
        query_retrasos = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'RETRASO%'
        """)
        total_retrasos = db.execute(query_retrasos).scalar_one()

        # Obtener total de salidas
        query_salidas = text("""
            SELECT COUNT(*) 
            FROM historial_accesos 
            WHERE estado_registro LIKE 'SALIDA%'
        """)
        total_salidas = db.execute(query_salidas).scalar_one()

        # Obtener total de reportes
        query_reportes = text("""
            SELECT COUNT(*) 
            FROM reportes
        """)
        total_reportes = db.execute(query_reportes).scalar_one()

        # Obtener usuarios activos
        query_usuarios = text("""
            SELECT COUNT(*) 
            FROM personas 
            WHERE activo = TRUE
        """)
        usuarios_activos = db.execute(query_usuarios).scalar_one()

        return {
            "total_entradas": total_entradas,
            "total_retrasos": total_retrasos,
            "total_salidas": total_salidas,
            "total_reportes": total_reportes,
            "usuarios_activos": usuarios_activos,
            "ultima_actualizacion": datetime.now()
        }

    except Exception as e:
        logger.error(f"Error al obtener resumen: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al generar el resumen"
        )

@app.post("/historial/filtrar/")
def filtrar_historial(filtro: FiltroHistorial, db: Session = Depends(get_db)):
    """Endpoint para filtrar el historial de accesos"""
    try:
        query = text("""
            SELECT ha.*, p.nombre, p.apellido_paterno, p.apellido_materno, c.nombre_usuario
            FROM historial_accesos ha
            JOIN personas p ON ha.id_persona = p.id_persona
            JOIN cuentas c ON p.id_persona = c.id_persona
            WHERE 1=1
        """)
        params = {}

        # Aplicar filtros
        if filtro.nombre_usuario:
            query = text(f"{query.text} AND c.nombre_usuario LIKE :nombre_usuario")
            params["nombre_usuario"] = f"%{filtro.nombre_usuario}%"
        
        if filtro.fecha_inicio:
            query = text(f"{query.text} AND ha.fecha >= :fecha_inicio")
            params["fecha_inicio"] = filtro.fecha_inicio
        
        if filtro.fecha_fin:
            query = text(f"{query.text} AND ha.fecha <= :fecha_fin")
            params["fecha_fin"] = filtro.fecha_fin
        
        if filtro.estado_registro:
            query = text(f"{query.text} AND ha.estado_registro = :estado_registro")
            params["estado_registro"] = filtro.estado_registro

        query = text(f"{query.text} ORDER BY ha.fecha DESC")
        result = db.execute(query, params)
        
        historial = []
        for row in result:
            historial.append({
                "id_acceso": row.id_acceso,
                "nombre_completo": f"{row.nombre} {row.apellido_paterno} {row.apellido_materno or ''}".strip(),
                "nombre_usuario": row.nombre_usuario,
                "fecha": row.fecha,
                "resultado": row.resultado,
                "estado_registro": row.estado_registro,
                "dispositivo": row.id_dispositivo,
                "confianza": row.confianza
            })

        return {"historial": historial}

    except Exception as e:
        logger.error(f"Error al filtrar historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al filtrar el historial"
        )

@app.get("/usuarios/buscar/", response_model=List[UsuarioCompleto])
def buscar_usuarios(
    filtro: FiltroUsuario = Depends(),
    db: Session = Depends(get_db)
):
    """Endpoint para buscar usuarios por nombre, apellido o ambos, incluyendo los sin rol"""
    try:
        # Validar que al menos un criterio de búsqueda esté presente
        if not filtro.nombre and not filtro.apellido:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar al menos un nombre o apellido para buscar"
            )

        # Construir la consulta base con LEFT JOIN para roles
        query = text("""
            SELECT 
                p.id_persona,
                p.nombre,
                p.apellido_paterno,
                p.apellido_materno,
                p.telefono,
                p.correo_electronico,
                p.activo,
                c.nombre_usuario,
                r.nombre as rol
            FROM personas p
            JOIN cuentas c ON p.id_persona = c.id_persona
            LEFT JOIN roles r ON c.id_rol = r.id_rol  -- Cambiado a LEFT JOIN
            WHERE 1=1
        """)
        params = {}

        # Añadir condiciones de búsqueda (igual que antes)
        conditions = []
        if filtro.nombre:
            conditions.append("p.nombre ILIKE :nombre")
            params["nombre"] = f"%{filtro.nombre}%"
        
        if filtro.apellido:
            conditions.append("(p.apellido_paterno ILIKE :apellido OR p.apellido_materno ILIKE :apellido)")
            params["apellido"] = f"%{filtro.apellido}%"

        if conditions:
            query = text(f"{query.text} AND ({' OR '.join(conditions)})")

        # Añadir límite
        query = text(f"{query.text} ORDER BY p.nombre, p.apellido_paterno LIMIT :limit")
        params["limit"] = filtro.limit

        # Ejecutar consulta principal
        result_personas = db.execute(query, params)
        
        usuarios = []
        for persona in result_personas:
            # Obtener horario (si existe)
            query_horario = text("""
                SELECT hora_entrada, hora_salida, dias_laborales
                FROM horarios_persona 
                WHERE id_persona = :id_persona
            """)
            horario = db.execute(query_horario, {"id_persona": persona.id_persona}).fetchone()
            
            # Obtener últimos 10 accesos
            query_accesos = text("""
                SELECT 
                    ha.fecha,
                    ha.resultado,
                    ha.estado_registro,
                    ha.confianza,
                    d.nombre as dispositivo
                FROM historial_accesos ha
                LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
                WHERE ha.id_persona = :id_persona
                ORDER BY ha.fecha DESC
                LIMIT 10
            """)
            accesos_result = db.execute(query_accesos, {"id_persona": persona.id_persona})
            
            accesos = [{
                "fecha": acceso.fecha,
                "dispositivo": acceso.dispositivo,
                "resultado": acceso.resultado,
                "estado_registro": acceso.estado_registro,
                "confianza": acceso.confianza
            } for acceso in accesos_result]
            
            # Obtener total de accesos
            query_total_accesos = text("""
                SELECT COUNT(*) 
                FROM historial_accesos 
                WHERE id_persona = :id_persona
            """)
            total_accesos = db.execute(query_total_accesos, {"id_persona": persona.id_persona}).scalar_one()
            
            usuarios.append({
                "id_persona": persona.id_persona,
                "nombre": persona.nombre,
                "apellido_paterno": persona.apellido_paterno,
                "apellido_materno": persona.apellido_materno,
                "telefono": persona.telefono,
                "correo_electronico": persona.correo_electronico,
                "activo": persona.activo,
                "nombre_usuario": persona.nombre_usuario,
                "rol": persona.rol,  # Será None si no tiene rol asignado
                "hora_entrada": str(horario.hora_entrada) if horario else None,
                "hora_salida": str(horario.hora_salida) if horario else None,
                "dias_laborales": horario.dias_laborales if horario else None,
                "accesos": accesos,
                "total_accesos": total_accesos
            })

        return usuarios

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al buscar usuarios: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al buscar usuarios"
        )
        
@app.get("/reportes/", response_model=List[dict])
def obtener_reportes(
    filtro: FiltroReportes = Depends(), 
    db: Session = Depends(get_db)
):
    """Endpoint para obtener reportes filtrados por estado, tipo y severidad"""
    try:
        # Validar valores de los filtros contra los permitidos en la base de datos
        valid_estados = ['Abierto', 'En progreso', 'Resuelto', 'Cerrado']
        valid_tipos = ['Error del sistema', 'Fallo autenticación', 'Fallo de dispositivo', 
                      'Acceso no autorizado', 'Horario irregular', 'Otros']
        valid_severidades = ['Baja', 'Media', 'Alta', 'Crítica']

        if filtro.estado and filtro.estado not in valid_estados:
            raise HTTPException(
                status_code=400,
                detail=f"Estado no válido. Opciones: {', '.join(valid_estados)}"
            )
        
        if filtro.tipo_reporte and filtro.tipo_reporte not in valid_tipos:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de reporte no válido. Opciones: {', '.join(valid_tipos)}"
            )
        
        if filtro.severidad and filtro.severidad not in valid_severidades:
            raise HTTPException(
                status_code=400,
                detail=f"Severidad no válida. Opciones: {', '.join(valid_severidades)}"
            )

        # Construir la consulta SQL
        query = text("""
            SELECT 
                r.id_reporte,
                r.titulo,
                r.descripcion,
                r.tipo_reporte,
                r.severidad,
                r.estado,
                r.fecha_generacion,
                r.fecha_cierre,
                p.nombre || ' ' || p.apellido_paterno as persona,
                d.nombre as dispositivo,
                r.evidencias,
                r.etiquetas
            FROM reportes r
            LEFT JOIN historial_accesos ha ON r.id_acceso_relacionado = ha.id_acceso
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON r.id_dispositivo = d.id_dispositivo
            WHERE 1=1
        """)
        params = {}
        
        # Aplicar filtros
        if filtro.estado:
            query = text(f"{query.text} AND r.estado = :estado")
            params["estado"] = filtro.estado
        
        if filtro.tipo_reporte:
            query = text(f"{query.text} AND r.tipo_reporte = :tipo_reporte")
            params["tipo_reporte"] = filtro.tipo_reporte
        
        if filtro.severidad:
            query = text(f"{query.text} AND r.severidad = :severidad")
            params["severidad"] = filtro.severidad
        
        # Ordenar y limitar
        query = text(f"{query.text} ORDER BY r.fecha_generacion DESC LIMIT :limit")
        params["limit"] = filtro.limit
        
        # Ejecutar consulta
        result = db.execute(query, params)
        
        # Procesar resultados
        reportes = []
        for row in result:
            reporte = {
                "id": row.id_reporte,
                "titulo": row.titulo,
                "descripcion": row.descripcion,
                "tipo": row.tipo_reporte,
                "severidad": row.severidad,
                "estado": row.estado,
                "fecha_creacion": row.fecha_generacion,
                "fecha_cierre": row.fecha_cierre,
                "persona": row.persona,
                "dispositivo": row.dispositivo,
                "evidencias": row.evidencias or [],
                "etiquetas": json.loads(row.etiquetas) if row.etiquetas else {}
            }
            reportes.append(reporte)
        
        return reportes

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener reportes: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al obtener reportes"
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auth-api"}
