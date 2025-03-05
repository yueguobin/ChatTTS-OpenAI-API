import torch
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ChatTTS
import io
import soundfile as sf
import re
from loguru import logger
import uvicorn
import argparse

app = FastAPI()
chat = ChatTTS.Chat()
chat.load_models(source="huggingface", model_path="2noise/ChatTTS")
p = '/app/ChatTTS/assest/spk_stat.pt'
std, mean = torch.load(p).chunk(2)
rand_spk = torch.randn(768) * std + mean

voice_mapping = {
    "alloy": "4099", "echo": "2222", "fable": "6653",
    "onyx": "7869", "nova": "5099", "shimmer": "4099"
}

def generate_speech(input_text, voice, temperature=0.3, prompt=''):
    params_infer_code = {'spk_emb': rand_spk, 'temperature': temperature, 'top_P': 0.8, 'top_K': 20}
    params_refine_text = {'prompt': prompt}
    torch.manual_seed(int(voice))
    wavs = chat.infer([input_text], use_decoder=True, params_infer_code=params_infer_code, params_refine_text=params_refine_text)
    return wavs[0][0]

def replace_non_alphanumeric(text):
    return re.sub(r'[^\w\s]', ' ', text)

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = 'alloy'
    response_format: str = 'wav'
    speed: float = 1.0
    temperature: float = 0.3
    prompt: str = '[oral_2][laugh_0][break_6]'

@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def create_speech(request: SpeechRequest, authorization: str = Header(...)):
    api_key = authorization.replace("Bearer ", "")
    if api_key != "your_secret_key":  # 替换为实际密钥
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not request.model or not request.input:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    try:
        input_text = replace_non_alphanumeric(request.input)
        voice = voice_mapping.get(request.voice, '4099')
        temperature = request.temperature
        prompt = request.prompt
        logger.info(f"[tts] input={input_text}, voice={voice}, temperature={temperature}")

        def audio_stream():
            for chunk in input_text.split(","):  # 可优化为按句子分块
                wav = generate_speech(chunk.strip(), voice, temperature, prompt)
                buffer = io.BytesIO()
                sf.write(buffer, wav, 24000, format="WAV")
                buffer.seek(0)
                yield buffer.read()

        return StreamingResponse(audio_stream(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)