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
    nombre: Optional[str] = Field(None, description="Filtrar por nombre. Usar 'NULL' para obtener registros sin persona asociada")
    apellido: Optional[str] = None
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None
    estado_registro: Optional[str] = None
    resultado: Optional[str] = None
    dispositivo_id: Optional[int] = None
    limit: Optional[int] = Field(10, ge=1, le=100)

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
    """Endpoint para filtrar el historial de accesos con todos los filtros opcionales (usando JSON)"""
    try:
        # Consulta base
        query = text("""
            SELECT 
                ha.id_acceso,
                ha.fecha,
                ha.resultado,
                ha.estado_registro,
                ha.confianza,
                ha.id_dispositivo,
                ha.razon,
                d.nombre as dispositivo_nombre,
                p.nombre,
                p.apellido_paterno,
                p.apellido_materno
            FROM historial_accesos ha
            LEFT JOIN personas p ON ha.id_persona = p.id_persona
            LEFT JOIN dispositivos d ON ha.id_dispositivo = d.id_dispositivo
            WHERE 1=1
        """)
        params = {}

        # Aplicar filtros (todos opcionales)
        if filtro.nombre is not None:  # Cambiado para aceptar explícitamente NULL
            if filtro.nombre == "NULL":  # Caso especial para filtrar por NULL
                query = text(f"{query.text} AND p.id_persona IS NULL")
            else:
                query = text(f"{query.text} AND p.nombre ILIKE :nombre")
                params["nombre"] = f"%{filtro.nombre}%"
        
        if filtro.apellido:
            query = text(f"{query.text} AND (p.apellido_paterno ILIKE :apellido OR p.apellido_materno ILIKE :apellido)")
            params["apellido"] = f"%{filtro.apellido}%"
        
        if filtro.fecha_inicio:
            query = text(f"{query.text} AND ha.fecha >= :fecha_inicio")
            params["fecha_inicio"] = filtro.fecha_inicio
        
        if filtro.fecha_fin:
            query = text(f"{query.text} AND ha.fecha <= :fecha_fin")
            params["fecha_fin"] = filtro.fecha_fin
        
        if filtro.estado_registro:
            query = text(f"{query.text} AND ha.estado_registro = :estado_registro")
            params["estado_registro"] = filtro.estado_registro
            
        if filtro.resultado:
            query = text(f"{query.text} AND ha.resultado = :resultado")
            params["resultado"] = filtro.resultado
            
        if filtro.dispositivo_id:
            query = text(f"{query.text} AND ha.id_dispositivo = :dispositivo_id")
            params["dispositivo_id"] = filtro.dispositivo_id

        # Ordenación y límite
        query = text(f"{query.text} ORDER BY ha.fecha DESC LIMIT :limit")
        params["limit"] = filtro.limit

        # Ejecutar consulta
        result = db.execute(query, params)
        
        # Procesar resultados
        historial = []
        for row in result:
            nombre_completo = ""
            if row.nombre:  # Solo construir nombre si existe
                nombre_completo = f"{row.nombre} {row.apellido_paterno} {row.apellido_materno or ''}".strip()
            
            historial.append({
                "id_acceso": row.id_acceso,
                "fecha": row.fecha,
                "nombre_completo": nombre_completo if nombre_completo else None,
                "resultado": row.resultado,
                "estado_registro": row.estado_registro,
                "confianza": row.confianza,
                "razon": row.razon,  # Añadido el campo razón
                "dispositivo": {
                    "id": row.id_dispositivo,
                    "nombre": row.dispositivo_nombre
                }
            })

        return {
            "total_resultados": len(historial),
            "historial": historial
        }

    except Exception as e:
        logger.error(f"Error al filtrar historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error interno al filtrar el historial"
        )
        
@app.post("/usuarios/buscar/")
def buscar_usuarios(
    filtro: FiltroUsuario, 
    db: Session = Depends(get_db)
):
    """Endpoint para buscar usuarios con nombre y/o apellido opcionales"""
    try:
        # Construir consulta base (ahora ambos filtros son opcionales)
        query = text("""
            SELECT 
                p.id_persona,
                p.nombre,
                p.apellido_paterno,
                p.apellido_materno,
                p.telefono,
                p.correo_electronico,
                p.activo,
                hp.hora_entrada,
                hp.hora_salida,
                hp.dias_laborales
            FROM personas p
            LEFT JOIN horarios_persona hp ON p.id_persona = hp.id_persona
            WHERE 1=1
        """)
        params = {}

        # Añadir condiciones de búsqueda (ambas opcionales)
        if filtro.nombre:
            query = text(f"{query.text} AND p.nombre ILIKE :nombre")
            params["nombre"] = f"%{filtro.nombre}%"
        
        if filtro.apellido:
            query = text(f"{query.text} AND (p.apellido_paterno ILIKE :apellido OR p.apellido_materno ILIKE :apellido)")
            params["apellido"] = f"%{filtro.apellido}%"

        # Añadir límite
        query = text(f"{query.text} ORDER BY p.nombre, p.apellido_paterno LIMIT :limit")
        params["limit"] = filtro.limit

        # Ejecutar consulta
        result = db.execute(query, params)
        
        usuarios = []
        for row in result:
            # Consultar estadísticas para cada usuario
            stats_query = text("""
                SELECT 
                    SUM(CASE WHEN estado_registro LIKE 'ENTRADA%' THEN 1 ELSE 0 END) as total_entradas,
                    SUM(CASE WHEN estado_registro LIKE 'RETRASO%' THEN 1 ELSE 0 END) as total_retrasos,
                    SUM(CASE WHEN estado_registro LIKE 'SALIDA%' THEN 1 ELSE 0 END) as total_salidas,
                    SUM(CASE WHEN estado_registro = 'FUERA_HORARIO' THEN 1 ELSE 0 END) as total_fuera_horario,
                    COALESCE(SUM(horas_extras), 0) as total_horas_extras
                FROM historial_accesos
                WHERE id_persona = :id_persona
            """)
            stats_result = db.execute(stats_query, {"id_persona": row.id_persona}).fetchone()
            
            usuarios.append({
                "id_persona": row.id_persona,
                "nombre": row.nombre,
                "apellido_paterno": row.apellido_paterno,
                "apellido_materno": row.apellido_materno,
                "telefono": row.telefono,
                "correo_electronico": row.correo_electronico,
                "activo": row.activo,
                "hora_entrada": str(row.hora_entrada) if row.hora_entrada else None,
                "hora_salida": str(row.hora_salida) if row.hora_salida else None,
                "dias_laborales": row.dias_laborales,
                "estadisticas": {
                    "total_entradas": stats_result.total_entradas or 0,
                    "total_retrasos": stats_result.total_retrasos or 0,
                    "total_salidas": stats_result.total_salidas or 0,
                    "total_fuera_horario": stats_result.total_fuera_horario or 0,
                    "total_horas_extras": float(stats_result.total_horas_extras or 0)
                }
            })

        return {
            "total_resultados": len(usuarios),
            "usuarios": usuarios
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en búsqueda de usuarios: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al buscar usuarios"
        )
        
@app.post("/reportes/filtrar/")
def filtrar_reportes(
    filtro: FiltroReportes, 
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
