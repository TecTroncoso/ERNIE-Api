import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import time
import re
import html 
import json
import asyncio
from gradio_client import Client, handle_file

# --- CONFIGURACIÓN ---
HF_SPACE_NAME = "baidu/ERNIE-4.5-VL-28B-A3B-Thinking"
# ---------------------

app = FastAPI()
client = None

def get_gradio_client():
    global client
    if client is None:
        print(f"🔄 Intentando conectar con Hugging Face Space: {HF_SPACE_NAME}...")
        try:
            client = Client(HF_SPACE_NAME)
            print("✅ Conectado exitosamente.")
        except Exception as e:
            print(f"❌ Error conectando al Space: {e}")
            raise HTTPException(status_code=503, detail=f"No se pudo conectar: {e}")
    return client

try:
    get_gradio_client()
except:
    pass 

# --- MODELOS DE DATOS ---
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    video_url: Optional[ImageUrl] = None 

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str = "ernie-vl"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# --- PROCESADO DE TEXTO MEJORADO ---
def format_ernie_response(raw_html: str) -> str:
    """
    Convierte el HTML de ERNIE a Markdown de forma segura para streaming.
    No espera a cerrar tags, simplemente reemplaza estructuras conocidas.
    """
    if not raw_html:
        return ""

    text = str(raw_html)
    
    # 1. Eliminar headers internos de ERNIE que solo dicen "Thinking Process" o "Answer"
    #    (Para evitar duplicados, ya que pondremos nuestros propios headers)
    text = re.sub(r'<div class="ernie-section-header">.*?</div>', '', text, flags=re.DOTALL)

    # 2. Reemplazar el inicio del bloque Thinking por nuestro header Markdown
    #    Detectamos la apertura del div principal de thinking
    if 'class="ernie-section ernie-thinking"' in text:
        text = text.replace('<div class="ernie-section ernie-thinking">', '🧠 **Thinking:**\n\n')
    
    # 3. Reemplazar el inicio del bloque Answer por un separador
    if 'class="ernie-section ernie-answer"' in text:
        text = text.replace('<div class="ernie-section ernie-answer">', '\n\n---\n\n')

    # 4. Decodificar entidades HTML (&quot;, &gt;, etc.)
    text = html.unescape(text)

    # 5. Eliminar TODAS las etiquetas HTML restantes (<div...>, </div>, <br>, etc.)
    #    Esto limpia el texto crudo.
    text = re.sub(r'<[^>]+>', '', text)

    return text.strip() # No hacemos strip() para no perder espacios finales importantes en stream, pero aquí ayuda a limpiar inicio.

# --- AUXILIAR HISTORIAL ---
def format_history_prompt(messages: List[Message]) -> str:
    history_text = ""
    for msg in messages[:-1]:
        role_name = "User" if msg.role == "user" else "Assistant"
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if item.type == "text":
                    content_str += (item.text or "") + " "
                elif item.type in ["image_url", "video_url"]:
                    content_str += "[Media file] "
        
        # Limpieza para no confundir al modelo con sus propios separadores anteriores
        content_str = content_str.replace("🧠 **Thinking:**", "").replace("---", "")
        
        if content_str.strip():
            history_text += f"{role_name}: {content_str}\n"
    
    if history_text:
        history_text = "--- Context ---\n" + history_text + "--- Request ---\n"
    return history_text

# --- GENERADOR DE STREAM MEJORADO ---
async def generate_stream(gradio_app, final_prompt, files_to_upload, model_name):
    job = gradio_app.submit(
        message={"text": final_prompt, "files": files_to_upload},
        api_name="/chat"
    )

    previous_text = ""
    chat_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    for result in job:
        raw_response = str(result[-1]) if isinstance(result, (list, tuple)) else str(result)
        
        current_formatted_text = format_ernie_response(raw_response)
        
        # --- PROTECCIÓN CONTRA JITTER ---
        # Si el texto nuevo es más corto que el anterior, es un glitch del parser HTML.
        # Ignoramos este frame y esperamos al siguiente donde el texto esté completo.
        if len(current_formatted_text) < len(previous_text):
            continue

        # Calculamos el delta solo si el texto ha crecido
        if len(current_formatted_text) > len(previous_text):
            # Solo enviamos la parte nueva
            delta_content = current_formatted_text[len(previous_text):]
            
            # Verificación extra: a veces el diff es solo un salto de línea basura
            if delta_content:
                previous_text = current_formatted_text
                
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta_content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.001) # Micro pausa para estabilidad

    # Chunk final
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    gradio_app = get_gradio_client()

    if not request.messages:
        raise HTTPException(status_code=400, detail="Mensajes vacíos")

    context_header = format_history_prompt(request.messages)
    last_msg = request.messages[-1]
    
    current_text = ""
    files_to_upload = []

    if isinstance(last_msg.content, str):
        current_text = last_msg.content
    elif isinstance(last_msg.content, list):
        for item in last_msg.content:
            if item.type == "text":
                current_text += (item.text or "") + " "
            elif item.type == "image_url" and item.image_url:
                files_to_upload.append(handle_file(item.image_url.url))
            elif item.type == "video_url" and item.video_url:
                print(f"🎥 Subiendo video: {item.video_url.url} ...")
                files_to_upload.append(handle_file(item.video_url.url))

    final_prompt = context_header + current_text

    if request.stream:
        print(f"🚀 Iniciando Stream ERNIE | Archivos: {len(files_to_upload)}")
        return StreamingResponse(
            generate_stream(gradio_app, final_prompt, files_to_upload, request.model),
            media_type="text/event-stream"
        )

    # Fallback No-Stream
    try:
        print(f"🚀 Enviando a ERNIE (No-Stream)...")
        result = gradio_app.predict(
            message={"text": final_prompt, "files": files_to_upload},
            api_name="/chat"
        )
        raw_response = str(result[-1]) if isinstance(result, (list, tuple)) else str(result)
        formatted_text = format_ernie_response(raw_response)
    except Exception as e:
        print(f"❌ Error: {e}")
        global client
        client = None 
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": formatted_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
