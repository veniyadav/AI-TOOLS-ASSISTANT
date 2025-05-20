import os
# import json
# import zipfile
import uuid
# import requests
import librosa  
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
# from langchain_groq import ChatGroq
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from dotenv import load_dotenv
import edge_tts
# from vosk import Model as VoskModel, KaldiRecognizer
# import wave
from prompt import system_prompt
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from globalllm import GroqLLM
import re
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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

# MODEL_DIR = "/app/models"
# MODEL_NAME = "vosk-model-en-in-0.5"
# MODEL_ZIP_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")
# MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"
processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en")
# --------------------------------------------------------------------------
# Download and Extract Model if Not Exists
# --------------------------------------------------------------------------
# def download_and_extract_model():
#     if not os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
#         os.makedirs(MODEL_DIR, exist_ok=True)
#         print("Downloading Vosk model...")
#         with requests.get(MODEL_URL, stream=True) as r:
#             r.raise_for_status()
#             with open(MODEL_ZIP_PATH, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)

#         print("Extracting Vosk model...")
#         with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
#             zip_ref.extractall(MODEL_DIR)

#         os.remove(MODEL_ZIP_PATH)
#         print("Vosk model downloaded and extracted to:", os.path.join(MODEL_DIR, MODEL_NAME))
#     else:
#         print("Model already exists at:", os.path.join(MODEL_DIR, MODEL_NAME))

# # --------------------------------------------------------------------------
# # Load Vosk Model
# # --------------------------------------------------------------------------
# download_and_extract_model()
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# vosk_model = VoskModel(MODEL_PATH)
# print("Vosk model loaded successfully from:", MODEL_PATH)


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
# async def voice_assistant_endpoint(request: Request, audio_file: UploadFile = File(...)):
#     if not audio_file:
#         raise HTTPException(status_code=400, detail="No audio file provided.")

#     # Create a unique temporary file name for the uploaded audio
#     temp_audio_path = f"temp_input_{uuid.uuid4().hex}.wav"
#     contents = await audio_file.read()
    
#     with open(temp_audio_path, "wb") as f:
#         f.write(contents)

#     try:
#         with wave.open(temp_audio_path, "rb") as wf:
#             channels = wf.getnchannels()
#             sample_width = wf.getsampwidth()
#             comp_type = wf.getcomptype()

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

#     try:
#         llm_response = generate_response(user_transcript)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")

#     # Check for navigation commands (URLs)
#     url_pattern = r'(https?://[^\s]+|/\w[\w/-]*)'
#     found_urls = re.findall(url_pattern, llm_response)
#     redirect_url = found_urls[0] if found_urls else None

#     # Save TTS audio
#     tts_filename = f"static/{uuid.uuid4().hex}.wav"
#     await text_to_speech(llm_response, tts_filename)

#     # ✅ Dynamically get full URL
#     base_url = str(request.base_url).rstrip("/")
#     audio_url = f"{base_url}/static/{os.path.basename(tts_filename)}"

#     os.remove(temp_audio_path)

#     return {
#         "transcription": user_transcript,
#         "response": llm_response,
#         "audio_url": audio_url,
#         "redirect_url": redirect_url if redirect_url else None
#     }
    
@app.post("/voice_assistant")
async def voice_assistant_endpoint(request: Request, audio_file: UploadFile = File(...)):
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    # Create a unique temporary file name for the uploaded audio
    temp_audio_path = f"temp_input_{uuid.uuid4().hex}.wav"
    contents = await audio_file.read()

    with open(temp_audio_path, "wb") as f:
        f.write(contents)

    try:
        audio, sr = librosa.load(temp_audio_path, sr=16000)
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

        # Access input_values as a dictionary key
        # input_values = audio_input.get("input_values")

        # Run the model
        with torch.no_grad():
            predicted_ids = model.generate(input_features=input_features)

        # Decode the result
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        user_transcript = transcription.strip()

        if not user_transcript:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    # Generate LLM response based on transcription
    try:
        llm_response = generate_response(user_transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM invocation error: {e}")
    
    # Check for navigation commands (URLs)
    url_pattern = r'(https?://[^\s]+|/\w[\w/-]*)'
    found_urls = re.findall(url_pattern, llm_response)
    redirect_url = found_urls[0] if found_urls else None

    # Save TTS audio
    tts_filename = f"static/{uuid.uuid4().hex}.wav"
    await text_to_speech(llm_response, tts_filename)

    # ✅ Dynamically get full URL
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/static/{os.path.basename(tts_filename)}"

    os.remove(temp_audio_path)

    return {
        "transcription": user_transcript,
        "response": llm_response,
        "audio_url": audio_url,
        "redirect_url": redirect_url if redirect_url else None
    }
# RUN THE APP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
