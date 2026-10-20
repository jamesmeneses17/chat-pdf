"""Página «Base de datos»: qué guarda PostgreSQL y cómo trabaja (solo lectura)."""
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para poder importar app/
from app import config  # noqa: E402

st.set_page_config(page_title="Base de datos · Base de Conocimiento IA", page_icon="🗄️", layout="wide")


def llamar(metodo: str, ruta: str, **kwargs):
    usuario = st.session_state.get("usuario_activo", "invitado")
    try:
        r = httpx.request(metodo, config.API_URL + ruta, headers={"X-Usuario": quote(usuario)}, timeout=60, **kwargs)
    except httpx.HTTPError:
        return None, "No se pudo conectar con la API. ¿Está corriendo uvicorn?"
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail", "Ocurrió un error.")
        except ValueError:
            return None, "Ocurrió un error en el servidor."
    return r.json(), None


def tabla(columnas: list[str], filas: list[list]):
    st.dataframe(pd.DataFrame(filas, columns=columnas), width="stretch", hide_index=True)


st.title("🗄️ Base de datos")
st.caption("Lo que guarda PostgreSQL (con pgvector) y cómo trabaja. Es solo lectura: aquí no se cambia nada. "
           f"Usuario: **{st.session_state.get('usuario_activo', 'invitado')}**")

datos, error = llamar("GET", "/db/resumen")
if error:
    st.error(error)
    st.stop()

c1, c2 = st.columns(2)
c1.metric("PostgreSQL", datos["version"])
c2.metric("Extensiones", ", ".join(datos["extensiones"]) or "—")

st.subheader("📚 Tablas")
tabla(["Tabla", "Filas", "Tamaño en disco", "Columnas"],
      [[t["nombre"], t["filas"], t["tamano"], ", ".join(t["columnas"])] for t in datos["tablas"]])
st.caption("«vector(768)» es el embedding: 768 números por fragmento o imagen, que permiten buscar por significado.")

st.subheader("📄 Tus documentos")
if datos["documentos"]:
    tabla(["Documento", "Modelo de embeddings", "Fragmentos", "Páginas con texto"],
          [[d["documento"], d["modelo_de_embeddings"], d["fragmentos"], d["paginas"]] for d in datos["documentos"]])
else:
    st.info("Aún no hay documentos de este usuario.")

st.subheader("⚡ Índices (lo que hace rápida la búsqueda)")
tabla(["Tabla", "Índice", "Definición"], [[i["tabla"], i["nombre"], i["definicion"]] for i in datos["indices"]])
st.caption("**hnsw** busca vectores parecidos sin comparar con todos (búsqueda por significado). "
           "**gin** acelera la búsqueda por palabras en español.")

st.subheader("🔍 Mirar una tabla")
nombres = [t["nombre"] for t in datos["tablas"]]
elegida = st.selectbox("Tabla", nombres, index=nombres.index("chunks") if "chunks" in nombres else 0)
muestra, error = llamar("GET", f"/db/muestra/{elegida}", params={"n": 10})
if error:
    st.error(error)
else:
    st.caption("Primeras 10 filas. Los vectores y archivos se muestran resumidos.")
    tabla(muestra["columnas"], muestra["filas"])

st.subheader("🧪 Consulta SQL (solo lectura)")
ejemplos = {
    "Fragmentos por documento": "SELECT documento, proveedor, COUNT(*) AS fragmentos FROM chunks GROUP BY 1, 2 ORDER BY 3 DESC",
    "Los 5 fragmentos más parecidos al primero": ("SELECT c.id, c.pagina, left(c.contenido, 80) AS texto, "
                                                "1 - (c.embedding <=> p.embedding) AS similitud "
                                                "FROM chunks c, (SELECT embedding FROM chunks ORDER BY id LIMIT 1) p "
                                                "WHERE c.proveedor = (SELECT proveedor FROM chunks ORDER BY id LIMIT 1) "
                                                "ORDER BY c.embedding <=> p.embedding LIMIT 5"),
    "Cómo busca por palabras (EXPLAIN)": ("EXPLAIN SELECT id FROM chunks "
                                          "WHERE to_tsvector('spanish', contenido) @@ to_tsquery('spanish', 'imagen')"),
    "Imágenes indexadas": "SELECT id, documento, pagina, ancho, alto FROM imagenes ORDER BY id",
}
ejemplo = st.selectbox("Ejemplo", list(ejemplos))
sql = st.text_area("Consulta", value=ejemplos[ejemplo], height=110, key=f"sql_{ejemplo}")
if st.button("▶ Ejecutar", type="primary"):
    resultado, error = llamar("POST", "/db/consulta", json={"sql": sql})
    if error:
        st.error(error)
    else:
        tabla(resultado["columnas"], resultado["filas"])
        st.caption(f"{len(resultado['filas'])} filas" + (" (recortado a 200)" if resultado["recortado"] else ""))
