#!/usr/bin/env python3
import os, base64, subprocess, cv2, pytesseract, tempfile
from openai import OpenAI

# 1) CONFIGURAÇÃO BÁSICA ------------------------------------------------
DEVICE = "/dev/v4l/by-id/usb-ICT-TEK_HD_Camera_202001010001-video-index0"
WIDTH, HEIGHT, FOURCC = 1920, 1080, "MJPG"        # resolução maior
MODEL_TEXT = "gpt-4o"
MODEL_TTS  = "tts-1"
VOICE      = "alloy"
ROTATE_180 = False
PROMPT_DESC = (
    "Descreva em português o que aparece na imagem. "
    "Se houver texto legível (livro, receita, capa, rótulo, etc.), "
    "adicionalmente termine com a linha isolada: TEXTO_PRESENTE=SIM. "
    "Caso contrário termine com TEXTO_PRESENTE=NAO."
)
PROMPT_OCR = (
    "Extraia todo o texto legível da imagem, mantendo quebras de linha. "
    "Responda apenas com o texto. Se não conseguir ler nada, "
    "responda 'SEM_TEXTO'."
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------------------------
def capture_frame() -> bytes:
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Falha na câmera")
    if ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes()

def vision(jpeg_b64: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{jpeg_b64}"}}
            ]
        }]
    )
    return resp.choices[0].message.content.strip()

def tts_play(text: str):
    speech = client.audio.speech.create(
        model=MODEL_TTS,
        voice=VOICE,
        input=text,
        response_format="wav"
    ).content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(speech)
        path = f.name
    subprocess.run(["aplay","-q","-D","plughw:2,0",path])

def ocr_tesseract(jpeg_bytes: bytes) -> str:
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, lang="eng+por").strip()

# ----------------------------------------------------------------------
def main():
    jpeg = capture_frame()
    b64  = base64.b64encode(jpeg).decode()

    # descrição
    desc = vision(b64, PROMPT_DESC)
    print("\n📷 DESCRIÇÃO:\n", desc)
    tts_play(desc)

    # flag
    if "TEXTO_PRESENTE=SIM" not in desc.upper():
        return

    # tentativa GPT para extração
    raw_text = vision(b64, PROMPT_OCR)
    if raw_text.strip().upper() == "SEM_TEXTO" or len(raw_text) < 20:
        print("\n⚠️  GPT não conseguiu ler; tentando Tesseract offline...")
        raw_text = ocr_tesseract(jpeg)

    if raw_text:
        print("\n📝 TEXTO LIDO:\n", raw_text)
        tts_play("Agora vou ler o texto encontrado: " + raw_text)
    else:
        print("\n🚫 Nenhum texto legível detectado.")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Defina OPENAI_API_KEY antes de rodar.")
    main()
