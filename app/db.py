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
        # Segundo en que empieza el fragmento (solo los que vienen de la transcripción de un audio o video)
        conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS inicio REAL")
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

        # --- Imágenes de los PDFs y su vector de Gemini (para buscar con una frase) ---
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS imagenes (
                id SERIAL PRIMARY KEY,
                usuario TEXT NOT NULL,
                documento TEXT NOT NULL,
                pagina INTEGER NOT NULL,
                ancho INTEGER,
                alto INTEGER,
                png BYTEA NOT NULL,
                embedding vector({config.GEMINI_DIM})
            )""")
        # La misma tabla guarda también audios y videos (por segmentos): tipo, cuándo empieza el segmento,
        # qué archivo es y con qué modelo se calculó el vector (los de modelos distintos no se comparan)
        for columna in ("tipo TEXT NOT NULL DEFAULT 'imagen'", "proveedor TEXT NOT NULL DEFAULT 'gemini'",
                        "inicio REAL", "fin REAL", "nombre TEXT"):
            conn.execute(f"ALTER TABLE imagenes ADD COLUMN IF NOT EXISTS {columna}")
        # Los archivos de audio y video originales (para reproducirlos), una sola vez por archivo
        conn.execute("""
            CREATE TABLE IF NOT EXISTS adjuntos (
                usuario TEXT NOT NULL,
                documento TEXT NOT NULL,
                nombre TEXT NOT NULL,
                tipo TEXT NOT NULL,
                mime TEXT NOT NULL,
                datos BYTEA NOT NULL,
                PRIMARY KEY (usuario, documento, nombre)
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS imagenes_usuario_idx ON imagenes (usuario, documento)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS imagenes_embedding_idx
            ON imagenes USING hnsw (embedding vector_cosine_ops)""")

        # --- Claves de Gemini: cuáles están encendidas y las que se pegaron desde la interfaz ---
        # (las del .env siguen funcionando; una clave guardada aquí tiene prioridad sobre la del .env)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gemini_claves (
                numero INTEGER PRIMARY KEY,
                activa BOOLEAN NOT NULL DEFAULT TRUE,
                clave TEXT NOT NULL DEFAULT ''
            )""")
        conn.execute("ALTER TABLE gemini_claves ADD COLUMN IF NOT EXISTS clave TEXT NOT NULL DEFAULT ''")

        # --- Ajustes generales (por ejemplo, qué modelo de embeddings usar al subir PDFs) ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ajustes (
                nombre TEXT PRIMARY KEY,
                valor TEXT NOT NULL
            )""")
        # Con qué modelo se calculó el vector de cada fragmento (los vectores de modelos distintos no se pueden comparar)
        conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS proveedor TEXT NOT NULL DEFAULT 'ollama'")

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
    """Devuelve los documentos del usuario con su cantidad de fragmentos y fecha de subida.

    Incluye los audios y videos sueltos (sin texto): su «tipo» es audio o video y sus fragmentos son segmentos.
    """
    with conectar() as conn:
        filas = conn.execute(
            """
            SELECT c.documento, COUNT(*), MAX(d.creado), MIN(c.proveedor),
                   (SELECT MIN(a.tipo) FROM adjuntos a
                     WHERE a.usuario = %s AND a.documento = c.documento AND a.nombre = c.documento)
            FROM chunks c
            LEFT JOIN documentos d ON d.usuario = c.usuario AND d.nombre = c.documento
            WHERE c.usuario = %s
            GROUP BY c.documento
            ORDER BY MAX(d.creado) DESC NULLS LAST, c.documento
            """,
            (usuario, usuario),
        ).fetchall()
        sueltos = conn.execute(
            """
            SELECT documento, MIN(tipo), COUNT(*), MIN(proveedor) FROM imagenes
            WHERE usuario = %s AND tipo IN ('audio', 'video')
              AND documento NOT IN (SELECT documento FROM chunks WHERE usuario = %s)
            GROUP BY documento ORDER BY documento
            """,
            (usuario, usuario),
        ).fetchall()
    docs = [{"nombre": n, "fragmentos": t, "creado": f.isoformat() if f else None, "embeddings": p,
             "tipo": tipo or "pdf"} for n, t, f, p, tipo in filas]
    docs += [{"nombre": n, "fragmentos": t, "creado": None, "embeddings": p, "tipo": tipo}
             for n, tipo, t, p in sueltos]
    return docs


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
        borradas = conn.execute("DELETE FROM imagenes WHERE usuario = %s AND documento = %s",
                                (usuario, nombre)).rowcount
        conn.execute("DELETE FROM adjuntos WHERE usuario = %s AND documento = %s", (usuario, nombre))
        return cur.rowcount + borradas  # un audio o video suelto no tiene fragmentos de texto


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


def guardar_imagenes(usuario: str, documento: str, filas: list[dict], adjuntos: list[dict] | None = None):
    """Reemplaza las imágenes / audios / videos indexados de un documento.

    Cada fila: {tipo, pagina, ancho, alto, png, embedding, proveedor, inicio, fin, nombre}.
    `adjuntos`: archivos originales de audio y video [{nombre, tipo, mime, datos}] para poder reproducirlos.
    """
    with conectar() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM imagenes WHERE usuario = %s AND documento = %s", (usuario, documento))
            conn.execute("DELETE FROM adjuntos WHERE usuario = %s AND documento = %s", (usuario, documento))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO imagenes (usuario, documento, tipo, pagina, ancho, alto, png, embedding, "
                    "proveedor, inicio, fin, nombre) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [(usuario, documento, f.get("tipo", "imagen"), f.get("pagina", 0), f.get("ancho"), f.get("alto"),
                      f.get("png", b""), f["embedding"], f.get("proveedor", "gemini"), f.get("inicio"),
                      f.get("fin"), f.get("nombre")) for f in filas],
                )
                cur.executemany(
                    "INSERT INTO adjuntos (usuario, documento, nombre, tipo, mime, datos) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    [(usuario, documento, a["nombre"], a["tipo"], a["mime"], a["datos"]) for a in adjuntos or []],
                )


def buscar_imagenes(usuario: str, vector, documentos: list[str] | None, n: int,
                    proveedor: str = "gemini", tipos: tuple = ("imagen",)) -> list[dict]:
    """Los `n` elementos (imágenes, audios, videos) del usuario más parecidos al vector (distancia coseno).

    Solo se comparan con los que se calcularon con el mismo `proveedor` que el vector.
    """
    filtro, extra = "usuario = %s AND proveedor = %s AND tipo = ANY(%s)", [usuario, proveedor, list(tipos)]
    if documentos:
        filtro += " AND documento = ANY(%s)"
        extra.append(documentos)
    with conectar() as conn:
        filas = conn.execute(
            f"SELECT id, documento, pagina, 1 - (embedding <=> %s) AS similitud, tipo, inicio, fin, nombre "
            f"FROM imagenes WHERE {filtro} ORDER BY embedding <=> %s LIMIT %s",
            (vector, *extra, vector, n),
        ).fetchall()
    return [{"id": i, "documento": d, "pagina": p, "similitud": float(s), "tipo": t, "inicio": ini, "fin": fin,
             "nombre": nom} for i, d, p, s, t, ini, fin, nom in filas]


def imagenes_de_documentos(usuario: str, documentos: list[str] | None, pagina: int | None = None,
                           limite: int = 3) -> list[dict]:
    """Imágenes indexadas (con sus bytes) de los documentos, opcionalmente solo de una página."""
    filtro, extra = "usuario = %s AND tipo = 'imagen'", [usuario]
    if documentos:
        filtro += " AND documento = ANY(%s)"
        extra.append(documentos)
    if pagina:
        filtro += " AND pagina = %s"
        extra.append(pagina)
    with conectar() as conn:
        filas = conn.execute(f"SELECT id, documento, pagina, png FROM imagenes WHERE {filtro} "
                             f"ORDER BY documento, pagina, id LIMIT %s", (*extra, limite)).fetchall()
    return [{"id": i, "documento": d, "pagina": p, "png": bytes(png)} for i, d, p, png in filas]


def proveedores_de_medios(usuario: str, tipos: tuple) -> list[str]:
    """Con qué modelos se calcularon los elementos indexados del usuario (para convertir la pregunta igual)."""
    with conectar() as conn:
        return [p for (p,) in conn.execute(
            "SELECT DISTINCT proveedor FROM imagenes WHERE usuario = %s AND tipo = ANY(%s)", (usuario, list(tipos)))]


def obtener_adjunto(usuario: str, imagen_id: int) -> tuple[str, bytes] | None:
    """(tipo MIME, bytes) del archivo de audio o video al que pertenece un segmento indexado."""
    with conectar() as conn:
        fila = conn.execute(
            "SELECT a.mime, a.datos FROM imagenes i JOIN adjuntos a "
            "ON a.usuario = i.usuario AND a.documento = i.documento AND a.nombre = i.nombre "
            "WHERE i.id = %s AND i.usuario = %s", (imagen_id, usuario)).fetchone()
    return (fila[0], bytes(fila[1])) if fila else None


def inicios_por_pagina(usuario: str, documento: str) -> dict[int, float]:
    """{número de fragmento: segundo en que empieza} de un audio o video transcrito (vacío para un PDF)."""
    with conectar() as conn:
        return {p: i for p, i in conn.execute(
            "SELECT pagina, MIN(inicio) FROM chunks WHERE usuario = %s AND documento = %s AND inicio IS NOT NULL "
            "GROUP BY pagina", (usuario, documento))}


def obtener_adjunto_de_documento(usuario: str, documento: str) -> tuple[str, str, bytes] | None:
    """(tipo, tipo MIME, bytes) del audio o video de un documento suelto (el primero que tenga)."""
    with conectar() as conn:
        fila = conn.execute("SELECT tipo, mime, datos FROM adjuntos WHERE usuario = %s AND documento = %s "
                            "ORDER BY nombre LIMIT 1", (usuario, documento)).fetchone()
    return (fila[0], fila[1], bytes(fila[2])) if fila else None


def claves_gemini_activas() -> dict[int, bool]:
    """{número de clave: encendida}. Las que no están registradas se consideran encendidas."""
    with conectar() as conn:
        return {n: a for n, a in conn.execute("SELECT numero, activa FROM gemini_claves").fetchall()}


def claves_gemini_guardadas() -> dict[int, str]:
    """{número: clave} de las claves pegadas desde la interfaz."""
    with conectar() as conn:
        return {n: k for n, k in conn.execute("SELECT numero, clave FROM gemini_claves").fetchall() if k}


def guardar_clave_gemini(numero: int, clave: str):
    """Guarda (o borra, si va vacía) la clave de un número."""
    with conectar() as conn:
        conn.execute(
            "INSERT INTO gemini_claves (numero, clave) VALUES (%s, %s) "
            "ON CONFLICT (numero) DO UPDATE SET clave = EXCLUDED.clave", (numero, clave))


def obtener_ajuste(nombre: str, defecto: str) -> str:
    with conectar() as conn:
        fila = conn.execute("SELECT valor FROM ajustes WHERE nombre = %s", (nombre,)).fetchone()
    return fila[0] if fila else defecto


def guardar_ajuste(nombre: str, valor: str):
    with conectar() as conn:
        conn.execute("INSERT INTO ajustes (nombre, valor) VALUES (%s, %s) "
                     "ON CONFLICT (nombre) DO UPDATE SET valor = EXCLUDED.valor", (nombre, valor))


def fijar_clave_gemini(numero: int, activa: bool):
    with conectar() as conn:
        conn.execute(
            "INSERT INTO gemini_claves (numero, activa) VALUES (%s, %s) "
            "ON CONFLICT (numero) DO UPDATE SET activa = EXCLUDED.activa", (numero, activa))


def obtener_imagen(usuario: str, imagen_id: int) -> bytes | None:
    with conectar() as conn:
        fila = conn.execute("SELECT png FROM imagenes WHERE id = %s AND usuario = %s",
                            (imagen_id, usuario)).fetchone()
    return bytes(fila[0]) if fila else None


def contar_imagenes_indexadas(usuario: str, documentos: list[str] | None = None) -> int:
    filtro, extra = "usuario = %s", [usuario]
    if documentos:
        filtro += " AND documento = ANY(%s)"
        extra.append(documentos)
    with conectar() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM imagenes WHERE {filtro}", extra).fetchone()[0]


def obtener_fragmentos_con_pagina(usuario: str, nombre: str) -> list[tuple[int, str]]:
    """(página, texto) de cada fragmento de un documento, en el orden del PDF."""
    with conectar() as conn:
        return conn.execute(
            "SELECT pagina, contenido FROM chunks WHERE usuario = %s AND documento = %s ORDER BY id",
            (usuario, nombre),
        ).fetchall()


# ------------------------------------------------------------ conversaciones
def crear_conversacion(usuario: str, titulo: str | None = None) -> int:
    with conectar() as conn:
        return conn.execute(
            "INSERT INTO conversaciones (usuario, titulo) VALUES (%s, %s) RETURNING id",
            (usuario, titulo),
        ).fetchone()[0]


def tipos_de_documentos(usuario: str) -> dict[str, str]:
    """{nombre: 'audio' | 'video'} de los archivos sueltos del usuario (los PDFs no aparecen: son el caso normal)."""
    with conectar() as conn:
        return {d: t for d, t in conn.execute(
            "SELECT documento, MIN(tipo) FROM adjuntos WHERE usuario = %s AND nombre = documento GROUP BY documento",
            (usuario,))}


def _documentos_del_mensaje(extra: str | None) -> list[str]:
    """Qué documentos usó una respuesta guardada (los marcados, o los que citó)."""
    try:
        datos = json.loads(extra or "{}")
    except ValueError:
        return []
    if datos.get("documentos_usados"):
        return list(datos["documentos_usados"])
    return sorted({f["documento"] for f in (datos.get("fuentes") or []) if f.get("documento")})


def listar_conversaciones(usuario: str) -> list[dict]:
    """Conversaciones del usuario, con los documentos que usó cada una (para mostrar de qué archivo trata)."""
    with conectar() as conn:
        filas = conn.execute(
            "SELECT id, titulo, actualizado FROM conversaciones WHERE usuario = %s "
            "ORDER BY actualizado DESC, id DESC",
            (usuario,),
        ).fetchall()
        ultimos = {}
        if filas:  # la última respuesta de cada conversación dice con qué documentos se hablaba
            ultimos = {c: e for c, e in conn.execute(
                "SELECT DISTINCT ON (conversacion_id) conversacion_id, extra FROM mensajes "
                "WHERE rol = 'assistant' AND conversacion_id = ANY(%s) ORDER BY conversacion_id, id DESC",
                ([f[0] for f in filas],))}
    return [{"id": i, "titulo": t or "Conversación nueva", "actualizado": a.isoformat(),
             "documentos": _documentos_del_mensaje(ultimos.get(i))}
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
