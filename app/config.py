"""Lee todas las variables de configuración desde el archivo .env.

Ningún otro archivo debe escribir estos valores a mano: todos importan de aquí.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# El .env está en la carpeta raíz del proyecto (un nivel arriba de app/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_MODE = os.getenv("AI_MODE", "mock").lower()  # "mock" u "ollama"
DATABASE_URL = os.getenv("DATABASE_URL", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
NUM_CTX = int(os.getenv("NUM_CTX", "2048"))  # tamaño de contexto del modelo (menos = más rápido)
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "10m")  # cuánto tiempo Ollama deja el modelo en RAM
BUSQUEDA_CANDIDATOS = int(os.getenv("BUSQUEDA_CANDIDATOS", "20"))  # fragmentos por vectores antes de reordenar
PESO_PALABRAS = float(os.getenv("PESO_PALABRAS", "0.15"))  # puntos extra por coincidir con las palabras de la pregunta
PESO_FTS = float(os.getenv("PESO_FTS", "0.1"))  # puntos extra por la búsqueda de texto completo
PESO_RRF = float(os.getenv("PESO_RRF", "0.1"))  # puntos extra por la fusión de las dos búsquedas
MARCAS_EN_TEXTO = os.getenv("MARCAS_EN_TEXTO", "1") == "1"  # poner [MARCADA] dentro del texto de la página
REPEAT_PENALTY =float(os.getenv("REPEAT_PENALTY", "1.15"))  # castigo a repetir palabras (1.0 = ninguno)
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "300"))  # máximo de tokens por respuesta (evita respuestas eternas)
TEMPERATURE =float(os.getenv("TEMPERATURE", "0.2"))  # menos = respuestas más fieles al texto
RESUMEN_CHARS =int(os.getenv("RESUMEN_CHARS", "3500"))  # texto máximo que se resume
OCR_LANG =os.getenv("OCR_LANG", "spa")  # idioma del OCR para PDFs escaneados
OCR_DPI = int(os.getenv("OCR_DPI", "200"))  # resolución con la que se "lee" la imagen
TESSDATA_DIR = os.getenv("TESSDATA_DIR", "")  # carpeta con los idiomas de Tesseract
