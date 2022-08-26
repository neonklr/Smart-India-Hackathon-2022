# ---------------------------- MODEL_RUNNER CONSTANTS ---------------------------- #

MODEL_PATHS = {
    'STT-online-v1': 'addy88/wav2vec2-sanskrit-stt',
    'Quant-heroku': '/app/Models/vakyansh-wav2vec2-sanskrit-sam-60-quantized/vakyansh-wav2vec2-sanskrit-sam-60_quant.pt',

    'STT-v1': 'Models/wav2vec2-sanskrit-stt',
    'STT-v2': 'Models/vakyansh-wav2vec2-sanskrit-sam-60',

    'Quant-v2': 'Models/vakyansh-wav2vec2-sanskrit-sam-60-quantized/vakyansh-wav2vec2-sanskrit-sam-60_quant.pt'
}

DEFAULT_MODEL_ID = 'STT-v2'
MODEL_AUDIO_RATE = 16000
QUANTIZED_MODEL_HEADER = "Quant"
BINARY_MODEL_HEADER = "STT"

# --------------------------- AUTOCORRECTOR CONSTANTS ---------------------------- #

MAX_ERROR = 3
MAX_WORDS_TO_PREDICT = 3

SANSKRIT_DICT_PATH = "Sanskrit Data/sanskrit_dictionary_ascii.json"


# ------------------------------ PREDICTOR CONSTANTS ----------------------------- #

CACHE_AUDIO_FILE_PATH = "Cache/audio"
AUDIO_FORMATS = {
    "audio/wav": "wav",
    "audio/webm": "webm"
}
FFMPEG_PATH = r"C:\Users\gupta\Desktop\ffmpeg-2022-08-22-git-f23e3ce858-full_build\bin\ffmpeg.exe"

# --------------------------------- OTHER CONSTANTS ------------------------------- #

DEBUGGING = False

# ---------------------------- TRANSLITERATOR CONSTANTS -------------------------- #

MAX_SUGGESTIONS = 6
