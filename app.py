import os
import json
import zipfile
import uuid
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from dotenv import load_dotenv
import edge_tts
from vosk import Model as VoskModel, KaldiRecognizer
import wave
from prompt import system_prompt
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from globalllm import GroqLLM
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static",follow_symlinks=True)
# Enable CORS for all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
# -----------------------------------------------------------------------------
# GLOBAL VARIABLES
# -----------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY")
groq_llm = GroqLLM(model="llama-3.3-70b-versatile", api_key=API_KEY,temperature=0.8)#llama-3.1-8b-instan

# Global dictionary for navigation commands.
navigation_commands = {}

# -----------------------------------------------------------------------------
# SETUP: Speech-to-Text (using Whisper) and Text-to-Speech (using Edge TTS)
# -----------------------------------------------------------------------------
# print("Loading vosk model and processor...")

MODEL_DIR = "/app/models"
MODEL_NAME = "vosk-model-en-in-0.5"
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"

# --------------------------------------------------------------------------
# Download and Extract Model if Not Exists
# --------------------------------------------------------------------------
def download_and_extract_model():
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading Vosk model...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_ZIP_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Extracting Vosk model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        os.remove(MODEL_ZIP_PATH)
        print("Vosk model downloaded and extracted to:", os.path.join(MODEL_DIR, MODEL_NAME))
    else:
        print("Model already exists at:", os.path.join(MODEL_DIR, MODEL_NAME))

# --------------------------------------------------------------------------
# Load Vosk Model
# --------------------------------------------------------------------------
download_and_extract_model()
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
vosk_model = VoskModel(MODEL_PATH)
print("Vosk model loaded successfully from:", MODEL_PATH)


async def text_to_speech(text: str, filename: str = "static/output.wav"):
    communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
    await communicate.save(filename)
    return filename

# -----------------------------------------------------------------------------
# LLM HELPER FUNCTION (ChatGroq)
# -----------------------------------------------------------------------------
def generate_response(user_text: str) -> str:
    # Create a ChatGroq client with the provided API key.
    # Build the combined system prompt:
    manual = system_prompt.get("manual") or ""
    url_comp = system_prompt.get("url") or ""
    manual_str = safe_str(manual)
    url_comp_str = safe_str(url_comp)
    combined_prompt = manual_str + "\n" + url_comp_str if manual_str and url_comp_str else manual_str or url_comp_str
    messages = [
        ("system", combined_prompt),
        ("human", user_text),
    ]
    ai_msg = groq_llm.invoke(messages)
    return ai_msg.content

def safe_str(val):
    return str(val) if val is not None else ""
# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------

# @app.post("/api/set_navigation_commands")
# async def set_navigation_commands_endpoint(data: dict):
#     global navigation_commands, system_prompt
#     if not data or not isinstance(data, dict):
#         raise HTTPException(status_code=400, detail="Provide a JSON object mapping command names to URLs.")
#     navigation_commands.update(data)
#     nav_list = [f"{key}: {url}" for key, url in navigation_commands.items()]
#     system_prompt["url"] = "Navigation Commands:\n" + "\n".join(nav_list)
#     return {
#         "message": "Navigation commands updated successfully.",
#         "navigation_commands": navigation_commands,
#         "system_prompt": system_prompt
#     }

@app.get("/api/get_prompt")
async def get_combined_prompt():
    manual = system_prompt.get("manual") or ""
    url_comp = system_prompt.get("url") or ""
    manual_str = safe_str(manual)
    url_comp_str = safe_str(url_comp)
    combined_prompt = manual_str + "\n" + url_comp_str if manual_str and url_comp_str else manual_str or url_comp_str
    if not combined_prompt:
        raise HTTPException(status_code=404, detail="No system prompt has been set yet.")
    return {"system_prompt": combined_prompt}

@app.post("/chat")
async def chat_endpoint(data: dict):
    if not data or "human_message" not in data:
        raise HTTPException(status_code=400, detail="Please provide a 'human_message' field.")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set. Use /api/set_api_key.")
    human_message = data["human_message"]
    try:
        ai_response = generate_response(human_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")
    
    url_pattern = r'(https?://[^\s]+|/\w[\w/-]*)'
    found_urls = re.findall(url_pattern, ai_response)

    result = {"response": ai_response}
    if found_urls:
        result["redirect_url"] = found_urls[0]  # You can use all if needed
    return result

# @app.post("/voice_assistant")
# async def voice_assistant_endpoint(audio_file: UploadFile = File(...)):
#     # print(audio_file)
#     if not audio_file:
#         raise HTTPException(status_code=400, detail="No audio file provided.")

#     temp_audio_path = "temp_input.wav"
#     contents = await audio_file.read()
    
#     with open(temp_audio_path, "wb") as f:
#         f.write(contents)

#     # Open the WAV file and check format
#     try:
#         with wave.open(temp_audio_path, "rb") as wf:
#             channels = wf.getnchannels()
#             sample_width = wf.getsampwidth()
#             comp_type = wf.getcomptype()

#             # Auto-convert to mono PCM 16-bit if needed
#             if channels != 1 or sample_width != 2 or comp_type != "NONE":
#                 raise HTTPException(status_code=400, detail="Audio must be 16-bit mono PCM WAV.")

#             rec = KaldiRecognizer(vosk_model, wf.getframerate())
#             result_text = ""

#             while True:
#                 data = wf.readframes(4000)
#                 if len(data) == 0:
#                     break
#                 if rec.AcceptWaveform(data):
#                     res = json.loads(rec.Result())
#                     result_text += res.get("text", "") + " "

#             final_res = json.loads(rec.FinalResult())
#             result_text += final_res.get("text", "")
#             user_transcript = result_text.strip()

#             if not user_transcript:
#                 raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

#     except wave.Error as e:
#         raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

#     # Generate LLM response
#     try:
#         llm_response = generate_response(user_transcript)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")

#     # Check for navigation commands
#     redirect_url = None
#     # url_pattern = r'(https?://[^\s]+/)'
#     # found_urls = re.findall(url_pattern, llm_response)
#      # Find both full URLs and relative routes
#     url_pattern = r'(https?://[^\s]+|/\w[\w/-]*)'
#     found_urls = re.findall(url_pattern, llm_response)
#     if found_urls:
#         redirect_url = found_urls[0]  # You can use all if needed


#     # Convert LLM response to speech
#     tts_filename = "static/output.wav"
#     await text_to_speech(llm_response, tts_filename)

#     result = {
#         "transcription": user_transcript,
#         "response": llm_response,
#         "audio_url": "/static/output.wav"
#     }
#     if redirect_url:
#         result["redirect_url"] = redirect_url
#     return result

@app.post("/voice_assistant")
async def voice_assistant_endpoint(audio_file: UploadFile = File(...)):
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # Create a unique temporary file name for the uploaded audio
    temp_audio_path = f"temp_input_{uuid.uuid4().hex}.wav"
    contents = await audio_file.read()
    
    with open(temp_audio_path, "wb") as f:
        f.write(contents)

    # Open the WAV file and check format
    try:
        with wave.open(temp_audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            comp_type = wf.getcomptype()

            # Auto-convert to mono PCM 16-bit if needed
            if channels != 1 or sample_width != 2 or comp_type != "NONE":
                raise HTTPException(status_code=400, detail="Audio must be 16-bit mono PCM WAV.")

            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            result_text = ""

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    result_text += res.get("text", "") + " "

            final_res = json.loads(rec.FinalResult())
            result_text += final_res.get("text", "")
            user_transcript = result_text.strip()

            if not user_transcript:
                raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

    except wave.Error as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    # Generate LLM response
    try:
        llm_response = generate_response(user_transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")

    # Check for navigation commands (URLs)
    redirect_url = None
    url_pattern = r'(https?://[^\s]+|/\w[\w/-]*)'
    found_urls = re.findall(url_pattern, llm_response)
    if found_urls:
        redirect_url = found_urls[0]  # You can use all if needed

    # Generate a unique file name for the TTS audio file
    tts_filename = f"static/{uuid.uuid4().hex}.wav"  # Unique file name for each request

    # Convert LLM response to speech
    await text_to_speech(llm_response, tts_filename)

    # Construct the public URL for the audio file
    BASE_URL = os.getenv("BASE_URL")  # Ensure your BASE_URL is set
    audio_url = f"{BASE_URL}/static/{os.path.basename(tts_filename)}"

    result = {
        "transcription": user_transcript,
        "response": llm_response,
        "audio_url": audio_url
    }
    
    if redirect_url:
        result["redirect_url"] = redirect_url
    
    # Clean up the temporary file
    os.remove(temp_audio_path)

    return result
# RUN THE APP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
