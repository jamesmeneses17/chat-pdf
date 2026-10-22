"""Lee todas las variables de configuración desde el archivo .env.

Ningún otro archivo debe escribir estos valores a mano: todos importan de aquí.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# El .env está en la carpeta raíz del proyecto (un nivel arriba de app/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_MODE = os.getenv("AI_MODE", "mock").lower()  # "mock" u "ollama"
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("@localhost", "@127.0.0.1")  # "localhost" prueba primero IPv6 (más lento)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
GEMMA2_MODEL = os.getenv("GEMMA2_MODEL", "google/embeddinggemma-2")  # EmbeddingGemma 2 (Hugging Face): texto, imágenes, audio, video
EMBED_GEMMA_MODEL = os.getenv("EMBED_GEMMA_MODEL", "embeddinggemma")  # opción local alternativa (768 números)
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
NUM_CTX = int(os.getenv("NUM_CTX", "3072"))  # tamaño de contexto del modelo (menos = más rápido)
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "10m")  # cuánto tiempo Ollama deja el modelo en RAM
BUSQUEDA_CANDIDATOS = int(os.getenv("BUSQUEDA_CANDIDATOS", "20"))  # fragmentos por vectores antes de reordenar
PESO_PALABRAS = float(os.getenv("PESO_PALABRAS", "0.15"))  # puntos extra por coincidir con las palabras de la pregunta
PESO_FTS = float(os.getenv("PESO_FTS", "0.1"))  # puntos extra por la búsqueda de texto completo
PESO_RRF = float(os.getenv("PESO_RRF", "0.1"))  # puntos extra por la fusión de las dos búsquedas
MARCAS_EN_TEXTO = os.getenv("MARCAS_EN_TEXTO", "1") == "1"  # poner [MARCADA] dentro del texto de la página
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.0"))  # castigo a repetir palabras; >1 perjudica al copiar del texto
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "300"))  # máximo de tokens por respuesta (evita respuestas eternas)
TEMPERATURE =float(os.getenv("TEMPERATURE", "0.2"))  # menos = respuestas más fieles al texto
RESUMEN_CHARS = int(os.getenv("RESUMEN_CHARS", "5000"))  # texto máximo que se resume (un fragmento por página)
OCR_LANG =os.getenv("OCR_LANG", "spa")  # idioma del OCR para PDFs escaneados
OCR_DPI = int(os.getenv("OCR_DPI", "200"))  # resolución con la que se "lee" la imagen
TESSDATA_DIR = os.getenv("TESSDATA_DIR", "")  # carpeta con los idiomas de Tesseract

# Gemini (nube), solo para buscar dentro de las imágenes de los PDFs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()  # vacía = función desactivada
# Hasta 3 claves, en orden de prioridad: la 1 es GEMINI_API_KEY; si una falla se pasa a la siguiente
GEMINI_CLAVES = [(1, GEMINI_API_KEY),
                 (2, os.getenv("GEMINI_API_KEY_2", "").strip()),
                 (3, os.getenv("GEMINI_API_KEY_3", "").strip())]
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
GEMINI_DIM = int(os.getenv("GEMINI_DIM", "768"))  # tamaño del vector de cada imagen
IMAGENES_MAX_POR_DOC = int(os.getenv("IMAGENES_MAX_POR_DOC", "40"))  # tope de imágenes enviadas por PDF
IMAGENES_TOP = int(os.getenv("IMAGENES_TOP", "6"))  # máximo de imágenes que se muestran por pregunta
# Qué imágenes cuentan como "coincidencia" (calibrado con puntajes reales de Gemini, ver README):
IMAGENES_MIN_SIM = float(os.getenv("IMAGENES_MIN_SIM", "0.585"))  # parecido mínimo (0 a 1)
IMAGENES_MARGEN = float(os.getenv("IMAGENES_MARGEN", "0.06"))  # solo las que están cerca de la mejor
# Audio y video (EmbeddingGemma 2): cada archivo se parte en segmentos para calcular sus vectores
SEGUNDOS_POR_SEGMENTO = int(os.getenv("SEGUNDOS_POR_SEGMENTO", "30"))
MEDIOS_MAX_SEGMENTOS = int(os.getenv("MEDIOS_MAX_SEGMENTOS", "20"))  # tope por archivo (20 x 30 s = 10 min)
VIDEO_SEGUNDOS_ENTRE_FOTOS = int(os.getenv("VIDEO_SEGUNDOS_ENTRE_FOTOS", "5"))  # cada fotograma tarda ~6 s en el CPU
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-base")  # transcribe audios y videos (base = rápido, small = más exacto)
WHISPER_IDIOMA = os.getenv("WHISPER_IDIOMA", "spanish")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")  # modelo de la nube que escribe respuestas / ve imágenes
# llama.cpp (llama-server): otra forma de ejecutar el modelo de chat local, sin Ollama de por medio
LLAMACPP_CARPETA = os.getenv("LLAMACPP_CARPETA", "llama.cpp")  # donde está llama-server.exe
LLAMACPP_PUERTO = int(os.getenv("LLAMACPP_PUERTO", "8090"))
LLAMACPP_MODELO = os.getenv("LLAMACPP_MODELO", "")  # archivo .gguf; vacío = el mismo de Ollama (CHAT_MODEL)
CARPETA_CONOCIMIENTO = os.getenv("CARPETA_CONOCIMIENTO", "conocimiento")  # carpeta con PDFs, audios y videos
IMAGENES_INCLUIR_ESCANEADAS = os.getenv("IMAGENES_INCLUIR_ESCANEADAS", "0") == "1"
# Parecido mínimo para aceptar una coincidencia y distancia máxima a la mejor, por modelo y tipo
# (calibrado con pruebas reales; cada modelo da números en escalas distintas)
UMBRALES = {
    "gemini": {"imagen": (IMAGENES_MIN_SIM, IMAGENES_MARGEN)},
    "gemma2": {"imagen": (0.655, 0.04), "audio": (0.69, 0.05), "video": (0.665, 0.04)},
}
