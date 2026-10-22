"""Trabajo en segundo plano: preguntas, subidas y resúmenes siguen aunque el usuario cambie de pantalla.

Streamlit vuelve a ejecutar la pantalla cada vez que se toca algo (abrir otro chat, marcar un archivo...). Lo que
se estaba haciendo DENTRO de esa ejecución se cortaba. Aquí cada trabajo largo corre en un hilo propio que habla con
la API y guarda su resultado en la base de datos, como cualquier mensaje de la conversación; la pantalla solo
muestra el avance y, al terminar, recarga lo guardado.

Este módulo no usa Streamlit (los hilos no pueden tocar `st.session_state`): todo lo que necesitan se les pasa.
"""
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx


@dataclass
class Tarea:
    id: int
    usuario: str
    tipo: str            # pregunta | resumen | puntos_clave | glosario | subida | carpeta | indexar
    titulo: str          # lo que se muestra: «Respondiendo…», «Subiendo Energia.pdf»
    conv_id: int | None = None
    documento: str | None = None
    estado: str = "corriendo"   # corriendo | lista | error
    texto: str = ""             # respuesta parcial que va llegando
    avance: str = ""            # qué está haciendo ahora
    inicio: float = field(default_factory=time.time)
    fin: float | None = None
    mensaje: str = ""           # resultado para mostrar cuando termina
    nivel: str = "info"         # success | warning | error | info
    avisos: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    visto: bool = False         # la pantalla ya la recogió (recargó la conversación)
    cerrada: bool = False       # el usuario la descartó del panel de actividad

    @property
    def segundos(self) -> int:
        return int((self.fin or time.time()) - self.inicio)


class Gestor:
    """Lista de tareas en marcha o recién terminadas. Una sola instancia para todo el servidor."""

    def __init__(self):
        self._tareas: dict[int, Tarea] = {}
        self._siguiente = itertools.count(1)
        self._candado = threading.Lock()

    def iniciar(self, usuario: str, tipo: str, titulo: str, trabajo, *, conv_id: int | None = None,
                documento: str | None = None) -> Tarea:
        """Arranca `trabajo(tarea)` en un hilo. El trabajo va llenando la tarea (texto, avance, mensaje...)."""
        with self._candado:
            tarea = Tarea(id=next(self._siguiente), usuario=usuario, tipo=tipo, titulo=titulo,
                          conv_id=conv_id, documento=documento)
            self._tareas[tarea.id] = tarea
            self._limpiar()
        threading.Thread(target=self._correr, args=(tarea, trabajo), daemon=True).start()
        return tarea

    @staticmethod
    def _correr(tarea: Tarea, trabajo):
        try:
            trabajo(tarea)
            if tarea.estado == "corriendo":
                tarea.estado = "lista"
        except Exception as e:  # un fallo inesperado no debe dejar la tarea «corriendo» para siempre
            tarea.estado, tarea.nivel = "error", "error"
            tarea.mensaje = f"Ocurrió un error inesperado: {e}"
        tarea.fin = time.time()

    def _limpiar(self):
        """Se olvidan las tareas terminadas hace más de una hora."""
        limite = time.time() - 3600
        for i in [i for i, t in self._tareas.items() if t.fin and t.fin < limite]:
            del self._tareas[i]

    def del_usuario(self, usuario: str) -> list[Tarea]:
        with self._candado:
            return sorted((t for t in self._tareas.values() if t.usuario == usuario and not t.cerrada),
                          key=lambda t: t.id)

    def corriendo(self, usuario: str, conv_id: int | None = None, tipos: tuple | None = None) -> list[Tarea]:
        return [t for t in self.del_usuario(usuario) if t.estado == "corriendo"
                and (conv_id is None or t.conv_id == conv_id) and (tipos is None or t.tipo in tipos)]

    def sin_recoger(self, usuario: str, conv_id: int | None, tipos: tuple) -> list[Tarea]:
        """Tareas de esta conversación que ya terminaron y la pantalla aún no recargó."""
        return [t for t in self.del_usuario(usuario) if t.estado != "corriendo" and not t.visto
                and t.conv_id == conv_id and t.tipo in tipos]

    def hay_actividad(self, usuario: str) -> bool:
        """True si hay algo corriendo o algo terminado que el usuario aún no vio."""
        return any(t.estado == "corriendo" or not t.visto for t in self.del_usuario(usuario))

    def cerrar(self, tarea_id: int):
        if tarea_id in self._tareas:
            self._tareas[tarea_id].cerrada = True


class Api:
    """Acceso a la API para los hilos de fondo."""

    def __init__(self, cliente: httpx.Client, base: str, cabeceras: dict):
        self.cliente, self.base, self.cab = cliente, base, cabeceras

    def llamar(self, metodo: str, ruta: str, **kwargs):
        """Devuelve (datos, error) como la función `llamar` de la interfaz."""
        try:
            r = self.cliente.request(metodo, self.base + ruta, headers=self.cab, timeout=900, **kwargs)
        except httpx.HTTPError:
            return None, "No se pudo conectar con la API. ¿Está corriendo uvicorn?"
        if r.status_code >= 400:
            try:
                return None, r.json().get("detail", "Ocurrió un error.")
            except ValueError:
                return None, "Ocurrió un error en el servidor."
        return r.json(), None

    def flujo(self, ruta: str, cuerpo: dict | None = None):
        """Respuestas por pedazos (una línea JSON cada una): entrega cada diccionario."""
        try:
            with self.cliente.stream("POST", self.base + ruta, json=cuerpo, headers=self.cab, timeout=900) as r:
                if r.status_code >= 400:
                    yield {"error": "Ocurrió un error en el servidor."}
                    return
                for linea in r.iter_lines():
                    if linea:
                        yield json.loads(linea)
        except httpx.HTTPError:
            yield {"error": "No se pudo conectar con la API. ¿Está corriendo uvicorn?"}


def guardar_mensaje(api: Api, conv_id: int, mensaje: dict):
    """Guarda un mensaje en la conversación (lo que no es rol ni texto va en «extra»)."""
    extra = {k: v for k, v in mensaje.items() if k not in ("rol", "texto")}
    api.llamar("POST", f"/conversaciones/{conv_id}/mensajes",
               json={"rol": mensaje["rol"], "texto": mensaje["texto"], "extra": extra})


# =============================================================================================== preguntas
def hacer_pregunta(api: Api, t: Tarea, *, pregunta: str, pregunta_enviada: str, documentos, historial: list,
                   documentos_usados: list):
    """Pregunta al chat, va dejando el texto parcial en la tarea y guarda la respuesta en la conversación."""
    t.avance = "Pensando"
    resultado, partes = {"fuentes": []}, []
    for dato in api.flujo("/chat_stream", {"pregunta": pregunta_enviada, "documentos": documentos,
                                           "historial": historial}):
        if "texto" in dato:
            partes.append(dato["texto"])
            t.texto, t.avance = "".join(partes), ""
        else:
            resultado.update(dato)
    if resultado.get("error"):
        guardar_mensaje(api, t.conv_id, {"rol": "assistant", "texto": f"No pude responder: {resultado['error']}"})
        t.estado, t.nivel, t.mensaje = "error", "error", resultado["error"]
        return
    t.meta["detalle"] = resultado.get("citas_detalle")
    guardar_mensaje(api, t.conv_id, {
        "rol": "assistant", "texto": "".join(partes), "fuentes": resultado["fuentes"],
        "citas": resultado.get("citas"), "citas_detalle": resultado.get("citas_detalle"),
        "total_buscado": resultado.get("total_buscado"), "tiempos": resultado.get("tiempos"),
        "modelo": resultado.get("modelo"), "interpretada": resultado.get("pregunta_interpretada"),
        "imagenes": resultado.get("imagenes"), "documentos_usados": documentos_usados, "pregunta": pregunta,
    })
    t.nivel, t.mensaje = "success", "Respuesta lista."


# =============================================================================================== resúmenes
def texto_de_cobertura(c: dict) -> str:
    """'Resumen basado en 12 de 42 fragmentos (uno por página, págs. 1–12), ≈ 38 % del texto.'"""
    if c.get("tipo", "pdf") != "pdf":  # audio o video: se resumió la transcripción
        return (f"Resumen basado en {c['fragmentos_usados']} de {c['fragmentos_total']} tramos de la "
                f"transcripción del {c['tipo']}, ≈ {c['porcentaje']} % del texto.")
    paginas = c["paginas_usadas"]
    if len(paginas) > 1 and paginas == list(range(paginas[0], paginas[-1] + 1)):
        lista = f"págs. {paginas[0]}–{paginas[-1]}"
    else:
        lista = "págs. " + ", ".join(str(p) for p in paginas)
    parcial = "" if c["paginas_usadas"] == c["paginas_total"] or len(paginas) == c["paginas_total"] \
        else f" de {c['paginas_total']} páginas"
    return (f"Resumen basado en {c['fragmentos_usados']} de {c['fragmentos_total']} fragmentos "
            f"(uno por página: {lista}{parcial}), ≈ {c['porcentaje']} % del texto.")


def hacer_resumen(api: Api, t: Tarea, nombre: str, modelo: str):
    """Genera el resumen de un archivo (con sus preguntas sugeridas) y lo guarda como mensaje del chat."""
    encabezado = f"**{nombre}** — resumen:"
    t.avance, t.texto = "Escribiendo el resumen", ""
    inicio, resultado, partes = time.time(), {}, []
    for dato in api.flujo(f"/documentos/{quote(nombre)}/resumen_stream"):
        if "texto" in dato:
            partes.append(dato["texto"])
            t.texto, t.avance = "".join(partes), ""
        else:
            resultado.update(dato)
    if resultado.get("error"):
        guardar_mensaje(api, t.conv_id, {"rol": "assistant", "documentos_usados": [nombre],
                                         "texto": f"{encabezado}\n\nNo pude generar el resumen: {resultado['error']}"})
        t.estado, t.nivel, t.mensaje = "error", "error", f"No pude generar el resumen de {nombre}: {resultado['error']}"
        return
    texto = "".join(partes)
    texto += f"\n\n*{modelo} · Resumen generado en {time.time() - inicio:.0f} s*"
    detalle = []
    cobertura = resultado.get("cobertura")
    if cobertura:  # cuántos fragmentos y páginas usó realmente el resumen
        texto += f"\n\n*{texto_de_cobertura(cobertura)}*"
        detalle = cobertura.get("detalle") or []
        if cobertura.get("cifras_sin_respaldo"):  # el modelo pudo inventar un número
            texto += ("\n\n*El resumen menciona cifras que no aparecen en el texto leído: "
                      + ", ".join(cobertura["cifras_sin_respaldo"]) + ". Verifícalas en el archivo.*")
    t.avance = "Preparando preguntas sugeridas"
    datos, _ = api.llamar("POST", f"/documentos/{quote(nombre)}/preguntas")
    guardar_mensaje(api, t.conv_id, {
        "rol": "assistant", "texto": f"{encabezado}\n\n{texto}", "sugerencias": datos["preguntas"] if datos else [],
        "citas_detalle": detalle, "documentos_usados": [nombre]})
    t.nivel, t.mensaje = "success", f"Resumen de {nombre} listo."


ETIQUETAS = {"puntos_clave": "puntos clave", "glosario": "glosario"}


def hacer_generacion(api: Api, t: Tarea, tipo: str, nombre: str, modelo: str):
    """Genera los puntos clave o el glosario y los guarda como mensaje del chat."""
    etiqueta = ETIQUETAS[tipo]
    encabezado = f"**{nombre}** — {etiqueta}:"
    t.avance = f"Generando {etiqueta}"
    inicio = time.time()
    datos, error = api.llamar("POST", f"/documentos/{quote(nombre)}/{tipo}")
    if error:
        guardar_mensaje(api, t.conv_id, {"rol": "assistant", "documentos_usados": [nombre],
                                         "texto": f"{encabezado}\n\nNo pude generar los {etiqueta}: {error}"})
        t.estado, t.nivel, t.mensaje = "error", "error", f"No pude generar los {etiqueta} de {nombre}: {error}"
        return
    guardar_mensaje(api, t.conv_id, {
        "rol": "assistant", "documentos_usados": [nombre],
        "texto": f"{encabezado}\n\n{datos[tipo]}\n\n*{modelo} · {time.time() - inicio:.0f} s*"})
    t.nivel, t.mensaje = "success", f"{etiqueta.capitalize()} de {nombre} listos."


# =============================================================================================== subidas
def hacer_subidas(api: Api, t: Tarea, archivos: list[tuple[str, bytes]], modelo: str):
    """Sube e indexa los archivos de uno en uno; de cada uno deja su resumen en la conversación."""
    ok, fallos, avisos = [], [], []
    for n, (nombre, contenido) in enumerate(archivos, 1):
        t.avance = f"Procesando {nombre} ({n} de {len(archivos)}): leyendo e indexando"
        t.documento = nombre
        datos, error = api.llamar("POST", "/subir", files={"archivo": (nombre, contenido, "application/octet-stream")})
        if error:
            fallos.append(f"{nombre}: {error}")
            continue
        ok.append(nombre)
        avisos += [f"{nombre}: {a}" for a in datos.get("advertencias", [])]
        t.meta["mensajes"] = t.meta.get("mensajes", []) + [datos["mensaje"]]
        if t.conv_id is not None:  # el resumen queda como primer mensaje de la conversación
            t.titulo = f"Resumiendo {nombre}"
            hacer_resumen(api, t, nombre, modelo)
            t.estado = "corriendo"  # hacer_resumen puede marcar error; la subida en sí salió bien
    t.avisos = avisos
    t.meta["subidos"] = ok
    if fallos and not ok:
        t.estado, t.nivel, t.mensaje = "error", "error", " · ".join(fallos)
    elif fallos:
        t.nivel, t.mensaje = "warning", f"Listos: {', '.join(ok)}. Fallaron: {' · '.join(fallos)}"
    else:
        t.nivel, t.mensaje = "success", ("Listo: " + ", ".join(ok))


# =============================================================================================== otras llamadas largas
def hacer_llamada(api: Api, t: Tarea, metodo: str, ruta: str, resumir):
    """Llamada larga a la API (indexar la carpeta, indexar imágenes...). `resumir(datos)` -> (mensaje, nivel)."""
    t.avance = t.titulo
    datos, error = api.llamar(metodo, ruta)
    if error:
        t.estado, t.nivel, t.mensaje = "error", "error", error
        return
    t.mensaje, t.nivel = resumir(datos)
