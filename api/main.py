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

# --- PROCESADO DE TEXTO ---
def format_ernie_response(raw_html: str) -> str:
    """
    Convierte el HTML sucio de ERNIE en texto Markdown limpio.
    """
    if not raw_html:
        return ""

    # Decodificar entidades HTML
    text = html.unescape(str(raw_html))
    
    # Intentar extraer Thinking y Answer con Regex
    # Nota: En stream, a veces los tags de cierre </div> no han llegado aún.
    # Usamos re.DOTALL para multilínea.
    
    thinking_match = re.search(r'class="ernie-section ernie-thinking".*?class="ernie-section-body">(.*?)</div>', text, re.DOTALL)
    answer_match = re.search(r'class="ernie-section ernie-answer".*?class="ernie-section-body">(.*?)</div>', text, re.DOTALL)
    
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    # Si encontramos estructura, formateamos
    if thinking_content or answer_content:
        final_output = ""
        if thinking_content:
            final_output += f"🧠 **Thinking:**\n\n{thinking_content}\n\n---\n\n"
        final_output += f"{answer_content}"
        
        # Si hay thinking pero no answer match (aún generando answer), intentamos sacar el resto
        if thinking_match and not answer_match:
            # Buscar si hay contenido después del thinking div
            end_thinking_index = thinking_match.end()
            remaining = text[end_thinking_index:]
            # Limpiar tags residuales del remaining
            clean_remaining = re.sub(r'<[^>]+>', '', remaining).strip()
            final_output += clean_remaining

        return final_output

    # Fallback: Si no hay estructura clara (o el stream está muy crudo), limpiar tags
    clean_text = re.sub(r'<[^>]+>', '', text) 
    return clean_text.strip()

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
        
        content_str = content_str.replace("🧠 **Thinking:**", "").replace("---", "")
        if content_str.strip():
            history_text += f"{role_name}: {content_str}\n"
    
    if history_text:
        history_text = "--- Context ---\n" + history_text + "--- Request ---\n"
    return history_text

# --- GENERADOR DE STREAM ---
async def generate_stream(gradio_app, final_prompt, files_to_upload, model_name):
    # Usamos submit() para obtener un Job que se puede iterar
    job = gradio_app.submit(
        message={"text": final_prompt, "files": files_to_upload},
        api_name="/chat"
    )

    previous_text = ""
    chat_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    # Iteramos sobre el generador del trabajo
    for result in job:
        # result suele ser una tupla/lista, el último elemento es el mensaje actual del bot
        raw_response = str(result[-1]) if isinstance(result, (list, tuple)) else str(result)
        
        # Limpiamos el HTML completo actual
        current_formatted_text = format_ernie_response(raw_response)
        
        # Calculamos solo la parte nueva (delta)
        if len(current_formatted_text) > len(previous_text):
            delta_content = current_formatted_text[len(previous_text):]
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
            
            # Pequeña pausa para no saturar el loop si el modelo va muy rápido
            await asyncio.sleep(0.01)

    # Chunk final para indicar fin
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

    # --- LÓGICA DE STREAMING ---
    if request.stream:
        print(f"🚀 Iniciando Stream ERNIE | Archivos: {len(files_to_upload)}")
        return StreamingResponse(
            generate_stream(gradio_app, final_prompt, files_to_upload, request.model),
            media_type="text/event-stream"
        )

    # --- LÓGICA SIN STREAMING (Original) ---
    try:
        print(f"🚀 Enviando a ERNIE (No-Stream) | Archivos: {len(files_to_upload)}")
        
        # predict() bloquea hasta el final
        result = gradio_app.predict(
            message={"text": final_prompt, "files": files_to_upload},
            api_name="/chat"
        )
        
        raw_response = str(result[-1]) if isinstance(result, (list, tuple)) else str(result)
        formatted_text = format_ernie_response(raw_response)

    except Exception as e:
        print(f"❌ Error en Gradio Predict: {e}")
        global client
        client = None 
        raise HTTPException(status_code=500, detail=f"Error del modelo: {str(e)}")

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
        "usage": {
            "prompt_tokens": len(final_prompt),
            "completion_tokens": len(formatted_text),
            "total_tokens": 0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
