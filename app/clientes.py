"""Clientes HTTP compartidos.

Crear un cliente httpx nuevo en cada llamada cuesta ~0,4 s en este computador (carga los certificados SSL
cada vez). Aquí se crean una sola vez y se reutilizan (también reutilizan la conexión abierta).
"""
import httpx

# Servicios que corren en este mismo computador (Ollama, llama.cpp): sin proxies ni certificados
LOCAL = httpx.Client(trust_env=False, verify=False)
# Servicios en internet (Gemini): sí se verifican los certificados
NUBE = httpx.Client()
