import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import time
import re
import html 
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

# --- PROCESADO DE TEXTO (NUEVO) ---
def format_ernie_response(raw_html: str) -> str:
    """
    Convierte el HTML sucio de ERNIE en texto Markdown limpio,
    conservando la sección de 'Thinking'.
    """
    # Decodificar entidades HTML (&quot; -> ", etc.)
    text = html.unescape(str(raw_html))
    
    # Extraer el contenido de Thinking
    thinking_match = re.search(r'class="ernie-section ernie-thinking".*?class="ernie-section-body">(.*?)</div>', text, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    
    # Extraer el contenido de la Respuesta (Answer)
    answer_match = re.search(r'class="ernie-section ernie-answer".*?class="ernie-section-body">(.*?)</div>', text, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""

    # Si no encontramos las etiquetas específicas, devolvemos el texto crudo limpio de tags básicos
    if not thinking_content and not answer_content:
        clean_text = re.sub(r'<[^>]+>', '', text) # Eliminar cualquier tag HTML
        return clean_text.strip()

    # Formatear salida final
    final_output = ""
    if thinking_content:
        final_output += f"🧠 **Thinking:**\n\n{thinking_content}\n\n---\n\n"
    
    final_output += f"{answer_content}"
    
    return final_output

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
        
        # Limpieza básica para no meter el output formateado de nuevo en el prompt
        content_str = content_str.replace("🧠 **Thinking:**", "").replace("---", "")
        
        if content_str.strip():
            history_text += f"{role_name}: {content_str}\n"
    
    if history_text:
        history_text = "--- Context ---\n" + history_text + "--- Request ---\n"
    return history_text

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

    try:
        print(f"🚀 Enviando a ERNIE | Archivos: {len(files_to_upload)}")
        
        result = gradio_app.predict(
            message={"text": final_prompt, "files": files_to_upload},
            api_name="/chat"
        )
        
        # Obtener el raw string (puede ser HTML)
        raw_response = str(result[-1]) if isinstance(result, (list, tuple)) else str(result)
        
        # Limpiar y formatear
        formatted_text = format_ernie_response(raw_response)

    except Exception as e:
        print(f"❌ Error en Gradio Predict: {e}")
        global client
        client = None # Forzar reconexión
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