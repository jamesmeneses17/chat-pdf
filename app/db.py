"""Conexión a PostgreSQL (pgvector) y operaciones sobre las tablas del proyecto.

Todo lo que guarda un usuario (documentos, fragmentos, conversaciones) queda separado
por el nombre de usuario: cada función recibe `usuario` y solo ve lo suyo.
"""
import json

import psycopg
from pgvector.psycopg import register_vector

from app import config

USUARIO_POR_DEFECTO = "invitado"


def conectar():
    """Abre una conexión nueva. Registra el tipo 'vector' si la extensión existe."""
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # La extensión aún no existe (primera vez); crear_tablas() la crea.
        pass
    return conn


def crear_tablas():
    """Crea las tablas e índices que falten y migra las bases creadas con versiones anteriores."""
    # Primero la extensión, con una conexión sin registrar el tipo vector
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    with conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                nombre TEXT PRIMARY KEY,
                creado TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")

        # --- Fragmentos de los PDFs (con su vector) ---
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                usuario TEXT NOT NULL DEFAULT '{USUARIO_POR_DEFECTO}',
                documento TEXT NOT NULL,
                pagina INTEGER NOT NULL,
                contenido TEXT NOT NULL,
                embedding vector({config.EMBED_DIM})
            )""")
        # Migración: bases anteriores no tenían la columna usuario
        conn.execute(
            f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS usuario TEXT NOT NULL "
            f"DEFAULT '{USUARIO_POR_DEFECTO}'")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_usuario_idx ON chunks (usuario, documento)")
        # Índice HNSW con distancia coseno: acelera la búsqueda por similitud
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)""")
        # Índice de texto completo (palabras clave, en español) para la búsqueda híbrida
        conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_fts_idx
            ON chunks USING GIN (to_tsvector('spanish', contenido))""")

        # --- El PDF original, su resumen, puntos clave y glosario ---
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documentos (
                usuario TEXT NOT NULL DEFAULT '{USUARIO_POR_DEFECTO}',
                nombre TEXT NOT NULL,
                pdf BYTEA NOT NULL,
                resumen TEXT,
                giros TEXT,
                puntos_clave TEXT,
                glosario TEXT,
                creado TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (usuario, nombre)
            )""")
        # Migración de bases anteriores (agrega las columnas que falten)
        conn.execute(
            f"ALTER TABLE documentos ADD COLUMN IF NOT EXISTS usuario TEXT NOT NULL "
            f"DEFAULT '{USUARIO_POR_DEFECTO}'")
        for columna in ("giros TEXT", "puntos_clave TEXT", "glosario TEXT",
                        "creado TIMESTAMPTZ NOT NULL DEFAULT now()"):
            conn.execute(f"ALTER TABLE documentos ADD COLUMN IF NOT EXISTS {columna}")
        # La clave primaria pasa de (nombre) a (usuario, nombre)
        conn.execute("""
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM pg_constraint
                         WHERE conrelid = 'documentos'::regclass AND contype = 'p'
                           AND array_length(conkey, 1) = 1) THEN
                ALTER TABLE documentos DROP CONSTRAINT documentos_pkey;
                ALTER TABLE documentos ADD PRIMARY KEY (usuario, nombre);
              END IF;
            END $$""")

        # --- Historial de conversaciones ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id SERIAL PRIMARY KEY,
                usuario TEXT NOT NULL,
                titulo TEXT,
                creado TIMESTAMPTZ NOT NULL DEFAULT now(),
                actualizado TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS conversaciones_usuario_idx
            ON conversaciones (usuario, actualizado DESC)""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mensajes (
                id SERIAL PRIMARY KEY,
                conversacion_id INTEGER NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
                rol TEXT NOT NULL,
                texto TEXT NOT NULL,
                extra TEXT,
                creado TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS mensajes_conv_idx ON mensajes (conversacion_id, id)")


# ---------------------------------------------------------------- usuarios
def asegurar_usuario(usuario: str):
    """Registra al usuario la primera vez que aparece."""
    with conectar() as conn:
        conn.execute("INSERT INTO usuarios (nombre) VALUES (%s) ON CONFLICT DO NOTHING", (usuario,))


# --------------------------------------------------------------- documentos
def listar_documentos(usuario: str) -> list[dict]:
    """Devuelve los documentos del usuario con su cantidad de fragmentos y fecha de subida."""
    with conectar() as conn:
        filas = conn.execute(
            """
            SELECT c.documento, COUNT(*), MAX(d.creado)
            FROM chunks c
            LEFT JOIN documentos d ON d.usuario = c.usuario AND d.nombre = c.documento
            WHERE c.usuario = %s
            GROUP BY c.documento
            ORDER BY MAX(d.creado) DESC NULLS LAST, c.documento
            """,
            (usuario,),
        ).fetchall()
    return [{"nombre": n, "fragmentos": t, "creado": f.isoformat() if f else None}
            for n, t, f in filas]


def documento_existe(usuario: str, nombre: str) -> bool:
    with conectar() as conn:
        fila = conn.execute(
            "SELECT 1 FROM chunks WHERE usuario = %s AND documento = %s LIMIT 1",
            (usuario, nombre),
        ).fetchone()
    return fila is not None


def borrar_documento(usuario: str, nombre: str) -> int:
    """Borra el documento con todos sus fragmentos, su PDF y su resumen.

    Devuelve cuántos fragmentos borró.
    """
    with conectar() as conn:
        cur = conn.execute("DELETE FROM chunks WHERE usuario = %s AND documento = %s",
                           (usuario, nombre))
        conn.execute("DELETE FROM documentos WHERE usuario = %s AND nombre = %s", (usuario, nombre))
        return cur.rowcount


def guardar_pdf(conn, usuario: str, nombre: str, contenido_pdf: bytes, giros: dict | None = None):
    """Guarda el PDF original (usa la conexión recibida, para ir en la misma transacción)."""
    conn.execute(
        "INSERT INTO documentos (usuario, nombre, pdf, giros) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (usuario, nombre) DO UPDATE SET pdf = EXCLUDED.pdf, giros = EXCLUDED.giros, "
        "resumen = NULL, puntos_clave = NULL, glosario = NULL",
        (usuario, nombre, contenido_pdf, json.dumps(giros or {})),
    )


def obtener_pdf(usuario: str, nombre: str) -> bytes | None:
    with conectar() as conn:
        fila = conn.execute("SELECT pdf FROM documentos WHERE usuario = %s AND nombre = %s",
                            (usuario, nombre)).fetchone()
    return bytes(fila[0]) if fila else None


def obtener_giros(usuario: str, nombre: str) -> dict:
    """Giros por página ({"3": 270, ...}) para mostrar derechas las hojas escaneadas de lado."""
    with conectar() as conn:
        fila = conn.execute("SELECT giros FROM documentos WHERE usuario = %s AND nombre = %s",
                            (usuario, nombre)).fetchone()
    return json.loads(fila[0]) if fila and fila[0] else {}


_CAMPOS_TEXTO = {"resumen", "puntos_clave", "glosario"}


def obtener_campo(usuario: str, nombre: str, campo: str) -> str | None:
    """Lee 'resumen', 'puntos_clave' o 'glosario' de un documento."""
    assert campo in _CAMPOS_TEXTO
    with conectar() as conn:
        fila = conn.execute(f"SELECT {campo} FROM documentos WHERE usuario = %s AND nombre = %s",
                            (usuario, nombre)).fetchone()
    return fila[0] if fila else None


def guardar_campo(usuario: str, nombre: str, campo: str, valor: str):
    assert campo in _CAMPOS_TEXTO
    with conectar() as conn:
        conn.execute(f"UPDATE documentos SET {campo} = %s WHERE usuario = %s AND nombre = %s",
                     (valor, usuario, nombre))


def obtener_fragmentos(usuario: str, nombre: str) -> list[str]:
    """Textos de los fragmentos de un documento, en el orden del PDF."""
    with conectar() as conn:
        filas = conn.execute(
            "SELECT contenido FROM chunks WHERE usuario = %s AND documento = %s ORDER BY id",
            (usuario, nombre),
        ).fetchall()
    return [f[0] for f in filas]


# ------------------------------------------------------------ conversaciones
def crear_conversacion(usuario: str, titulo: str | None = None) -> int:
    with conectar() as conn:
        return conn.execute(
            "INSERT INTO conversaciones (usuario, titulo) VALUES (%s, %s) RETURNING id",
            (usuario, titulo),
        ).fetchone()[0]


def listar_conversaciones(usuario: str) -> list[dict]:
    with conectar() as conn:
        filas = conn.execute(
            "SELECT id, titulo, actualizado FROM conversaciones WHERE usuario = %s "
            "ORDER BY actualizado DESC, id DESC",
            (usuario,),
        ).fetchall()
    return [{"id": i, "titulo": t or "Conversación nueva", "actualizado": a.isoformat()}
            for i, t, a in filas]


def conversacion_es_del_usuario(usuario: str, conversacion_id: int) -> bool:
    with conectar() as conn:
        return conn.execute("SELECT 1 FROM conversaciones WHERE id = %s AND usuario = %s",
                            (conversacion_id, usuario)).fetchone() is not None


def guardar_mensaje(conversacion_id: int, rol: str, texto: str, extra: dict | None = None):
    """Guarda un mensaje. El título de la conversación es la primera pregunta del usuario."""
    with conectar() as conn:
        conn.execute(
            "INSERT INTO mensajes (conversacion_id, rol, texto, extra) VALUES (%s, %s, %s, %s)",
            (conversacion_id, rol, texto, json.dumps(extra or {})),
        )
        titulo = (texto[:60] + "…") if len(texto) > 60 else texto
        conn.execute(
            "UPDATE conversaciones SET actualizado = now(), "
            "titulo = CASE WHEN titulo IS NULL AND %s = 'user' THEN %s ELSE titulo END "
            "WHERE id = %s",
            (rol, titulo, conversacion_id),
        )


def obtener_mensajes(conversacion_id: int) -> list[dict]:
    with conectar() as conn:
        filas = conn.execute(
            "SELECT rol, texto, extra FROM mensajes WHERE conversacion_id = %s ORDER BY id",
            (conversacion_id,),
        ).fetchall()
    return [{"rol": rol, "texto": texto, **(json.loads(extra) if extra else {})}
            for rol, texto, extra in filas]


def borrar_conversacion(usuario: str, conversacion_id: int) -> bool:
    with conectar() as conn:
        cur = conn.execute("DELETE FROM conversaciones WHERE id = %s AND usuario = %s",
                           (conversacion_id, usuario))
        return cur.rowcount > 0


def base_responde() -> bool:
    """True si la base de datos contesta una consulta simple."""
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
