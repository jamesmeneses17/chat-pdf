"""Consultas de solo lectura para mirar qué hay guardado en PostgreSQL y cómo trabaja (página «Base de datos»).

Nada de aquí modifica datos. La tabla de claves de Gemini no se puede consultar: sus claves nunca se muestran.
"""
import re

from app import db

TABLAS_OCULTAS = {"gemini_claves"}  # contiene las claves de API pegadas en la interfaz
MAX_FILAS = 200


def _legible(valor):
    """Convierte un valor de la base en algo corto y mostrable (los vectores y archivos se resumen)."""
    if valor is None:
        return None
    if isinstance(valor, (bytes, bytearray, memoryview)):
        return f"<binario, {len(bytes(valor)) / 1024:.0f} KB>"
    if hasattr(valor, "to_list"):  # vector de pgvector
        v = valor.to_list()
        return f"<vector de {len(v)} números: {', '.join(f'{x:.3f}' for x in v[:6])}, …>"
    if isinstance(valor, str) and len(valor) > 300:
        return valor[:300] + "…"
    return valor if isinstance(valor, (int, float, bool, str)) else str(valor)


def _filas(cursor) -> tuple[list[str], list[list]]:
    columnas = [c.name for c in cursor.description]
    return columnas, [[_legible(v) for v in fila] for fila in cursor.fetchall()]


def resumen(usuario: str) -> dict:
    """Versión, extensiones, tablas con su tamaño, índices y qué hay guardado del usuario."""
    with db.conectar() as conn:
        version = conn.execute("SHOW server_version").fetchone()[0]
        extensiones = [f"{n} {v}" for n, v in conn.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1")]
        tablas = []
        for (nombre,) in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1").fetchall():
            if nombre in TABLAS_OCULTAS:
                continue
            filas = conn.execute(f'SELECT COUNT(*) FROM "{nombre}"').fetchone()[0]
            tamano = conn.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (f'public."{nombre}"',)).fetchone()[0]
            columnas = [f"{c} ({t})" for c, t in conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position", (nombre,))]
            tablas.append({"nombre": nombre, "filas": filas, "tamano": tamano, "columnas": columnas})
        indices = [{"tabla": t, "nombre": n, "definicion": d} for t, n, d in conn.execute(
            "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
            "AND tablename <> ALL(%s) ORDER BY 1, 2", (list(TABLAS_OCULTAS),))]
        documentos = [{"documento": d, "modelo_de_embeddings": p, "fragmentos": n, "paginas": pag}
                      for d, p, n, pag in conn.execute(
                          "SELECT documento, proveedor, COUNT(*), COUNT(DISTINCT pagina) FROM chunks "
                          "WHERE usuario = %s GROUP BY documento, proveedor ORDER BY documento", (usuario,))]
    return {"version": version, "extensiones": extensiones, "tablas": tablas,
            "indices": indices, "documentos": documentos}


def muestra(tabla: str, n: int = 10) -> dict:
    """Las primeras filas de una tabla (los vectores y archivos se muestran resumidos)."""
    if tabla in TABLAS_OCULTAS:
        raise ValueError("Esa tabla guarda claves y no se puede mostrar.")
    with db.conectar() as conn:
        existentes = {t for (t,) in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        if tabla not in existentes:
            raise ValueError("Esa tabla no existe.")
        cur = conn.execute(f'SELECT * FROM "{tabla}" ORDER BY 1 LIMIT %s', (max(1, min(n, 50)),))
        columnas, filas = _filas(cur)
    return {"columnas": columnas, "filas": filas}


_SOLO_LECTURA = re.compile(r"^\s*(select|with|explain|show)\b", re.I)
_PROHIBIDO = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|do|vacuum)\b", re.I)


def consulta(sql: str) -> dict:
    """Ejecuta una consulta SELECT (o EXPLAIN) en modo solo lectura y devuelve hasta 200 filas."""
    sql = sql.strip().rstrip(";")
    if not _SOLO_LECTURA.match(sql) or ";" in sql or _PROHIBIDO.search(sql):
        raise ValueError("Solo se permiten consultas de lectura: SELECT, WITH, EXPLAIN o SHOW (una sola instrucción).")
    if re.search(r"gemini_claves", sql, re.I):
        raise ValueError("Esa tabla guarda claves y no se puede consultar.")
    with db.conectar() as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL statement_timeout = '15s'")
            cur = conn.execute(sql)
            if cur.description is None:
                return {"columnas": [], "filas": [], "recortado": False}
            columnas = [c.name for c in cur.description]
            filas = [[_legible(v) for v in f] for f in cur.fetchmany(MAX_FILAS + 1)]
    return {"columnas": columnas, "filas": filas[:MAX_FILAS], "recortado": len(filas) > MAX_FILAS}
