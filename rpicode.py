#!/usr/bin/env python3
"""
Raspberry Pi 4 B   —  Foto → descrição → OCR → fala → (rua / bairro / cidade)

• Dispara apenas quando a chave (SPDT) ligada ao GPIO 22 bascula  
• Descreve a cena (GPT-4o) e lê em voz alta  
• Se encontrar texto, tenta extrair (GPT-4o → fallback Tesseract) e lê  
• Escaneia Wi-Fi → Mozilla Location Service → Nominatim  
  – se falhar, usa IP-based lookup via ipinfo.io  
  – fala “Estamos na região de Rua X, Bairro Y, Cidade Z… precisão ±N m”

──────────────────────────
Requisitos (uma vez só):

sudo apt update
sudo apt install -y alsa-utils tesseract-ocr \
                    tesseract-ocr-por tesseract-ocr-eng \
                    python3-rpi.gpio iw sox
python3 -m venv ~/venvs/ai
source ~/venvs/ai/bin/activate
pip install --upgrade openai opencv-python numpy pytesseract requests
echo 'export OPENAI_API_KEY="sk-…"' >> ~/.bashrc

Adicione seu usuário ao grupo ‘gpio’ e faça logout/login:
sudo usermod -aG gpio $USER
──────────────────────────
Ligação da chave
• BCM 22 (pino fís. 15)  → comum  
• GND → polo A • 3 V3 → polo B
──────────────────────────
Execute (sem sudo):

source ~/venvs/ai/bin/activate
python ~/capture_describe.py
"""

import os, base64, subprocess, cv2, pytesseract, tempfile, time, signal, re, json
import RPi.GPIO as GPIO
import requests
from openai import OpenAI

# ╔═ IA / ÁUDIO / CÂMERA ════════════════════════════════════════════════
DEVICE  = "/dev/v4l/by-id/usb-ICT-TEK_HD_Camera_202001010001-video-index0"
WIDTH, HEIGHT, FOURCC = 1280, 720, "MJPG"
MODEL_TEXT = "gpt-4o"
MODEL_TTS  = "tts-1"
VOICE      = "alloy"
ROTATE_180 = False

# ╔═ GPIO POLLING ═══════════════════════════════════════════════════════
PIN_TOGGLE   = 22      # BCM 22 (fís. 15)
POLL_MS      = 40
DEBOUNCE_MS  = 200

# ╔═ PROMPTS ════════════════════════════════════════════════════════════
PROMPT_DESC = (
    "Descreva em português o que aparece na imagem. "
    "Se houver texto legível, acrescente a linha isolada TEXTO_PRESENTE=SIM, "
    "caso contrário TEXTO_PRESENTE=NAO."
)
PROMPT_OCR = (
    "Extraia todo o texto legível da imagem. "
    "Se não houver texto, responda apenas SEM_TEXTO."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ╔═ FUNÇÕES BÁSICAS ════════════════════════════════════════════════════
def tts_play(text: str):
    wav = client.audio.speech.create(
        model=MODEL_TTS, voice=VOICE, input=text, response_format="wav"
    ).content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav); path = f.name
    subprocess.run(["aplay", "-q", "-D", "plughw:2,0", path], check=True)

def capture_jpeg() -> bytes:
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("Falha na câmera")
    if ROTATE_180: frame = cv2.rotate(frame, cv2.ROTATE_180)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes()

# ╔═ WIFI → LOCALIZAÇÃO ═════════════════════════════════════════════════
def scan_wifi():
    raw = subprocess.check_output(["sudo","iw","dev","wlan0","scan","ap-force"])
    aps = []
    for cell in raw.decode(errors="ignore").split("BSS ")[1:]:
        mac = cell.split()[0]
        m   = re.search(r"signal:\s*(-\d+)", cell)
        if mac and m: aps.append({"macAddress":mac,"signalStrength":int(m.group(1))})
    return aps[:20]

def geo_from_mls():
    aps = scan_wifi()
    if not aps: return (None,None,None)
    try:
        url = "https://location.services.mozilla.com/v1/geolocate?key=test"
        loc = requests.post(url, json={"wifiAccessPoints": aps}, timeout=5).json()
        lat = loc["location"]["lat"]; lon = loc["location"]["lng"]
        return (lat, lon, loc.get("accuracy"))
    except Exception as e:
        print("⚠️ MLS:", e); return (None,None,None)

def geo_from_ip():
    try:
        j  = requests.get("https://ipinfo.io/json", timeout=4).json()
        lat, lon = j.get("loc","").split(",") if "loc" in j else (None,None)
        return (lat, lon, 50000)
    except Exception as e:
        print("⚠️ IPinfo:", e); return (None,None,None)

def reverse_nominatim(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format":"jsonv2","lat":lat,"lon":lon,"zoom":16}
        j = requests.get(url, params=params,
                         headers={"User-Agent":"rpi-cam/1.0"}).json()
        addr = j.get("address", {})
        rua   = addr.get("road") or ""
        bairro= addr.get("suburb") or addr.get("neighbourhood") or ""
        cidade= addr.get("city") or addr.get("town") or addr.get("village") or ""
        estado= addr.get("state") or ""
        return ", ".join(x for x in [rua,bairro,cidade,estado] if x)
    except Exception:
        return ""

# ╔═ OCR OFFLINE (fallback) ═════════════════════════════════════════════
def ocr_tesseract(jpeg: bytes):
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg,np.uint8), cv2.IMREAD_COLOR)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, lang="por+eng").strip()

# ╔═ PIPELINE P/ CADA TOGGLE ════════════════════════════════════════════
def process_once():
    jpeg = capture_jpeg(); b64 = base64.b64encode(jpeg).decode()

    desc = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"user","content":[
            {"type":"text","text":PROMPT_DESC},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]
    ).choices[0].message.content.strip()

    print("\n📷 DESCRIÇÃO:\n", desc)
    tts_play(desc)

    # ── texto?
    if "TEXTO_PRESENTE=SIM" in desc.upper():
        text = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[{"role":"user","content":[
                {"type":"text","text":PROMPT_OCR},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]
        ).choices[0].message.content.strip()

        if text.upper()=="SEM_TEXTO" or len(text)<20:
            print("⚠️  GPT não leu; Tesseract…")
            text = ocr_tesseract(jpeg)
        if text:
            print("\n📝 TEXTO LIDO:\n", text)
            tts_play("Lendo o texto encontrado: " + text)

    # ── localização
    lat, lon, acc = geo_from_mls()
    if not lat: lat, lon, acc = geo_from_ip()
    if lat:
        place = reverse_nominatim(lat, lon) or "local não identificado"
        msg = f"Estamos na região de {place}. Precisão aproximada {int(acc)} metros."
        print(f"\n📍 {place} (±{acc} m)"); tts_play(msg)
    else:
        print("\n📍 Localização indisponível.")

# ╔═ LOOP DE POLLING DO GPIO ─═══════════════════════════════════════════
def main():
    if not client.api_key: raise SystemExit("Defina OPENAI_API_KEY.")
    GPIO.setmode(GPIO.BCM); GPIO.setup(PIN_TOGGLE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    last = GPIO.input(PIN_TOGGLE); last_t = time.monotonic()
    print("Pronto! Aguardando chave no GPIO22… Ctrl+C para sair.")
    try:
        while True:
            time.sleep(POLL_MS/1000)
            state = GPIO.input(PIN_TOGGLE); now = time.monotonic()
            if state!=last and (now-last_t)*1000>DEBOUNCE_MS:
                print("\n🔔 Toggle detectado — iniciando…")
                process_once(); last_t = now
            last = state
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: GPIO.cleanup())
    main()
