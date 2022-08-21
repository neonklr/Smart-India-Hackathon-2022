# ---------------------------- MODEL RUNNER CONSTANTS ---------------------------- #

MODEL_PATHS = {
    'STT-online': 'addy88/wav2vec2-sanskrit-stt',
    'STT-heroku': '/app/Models/wav2vec2-sanskrit-stt',
    'STT-v1': 'Models/wav2vec2-sanskrit-stt'
}

DEFAULT_MODEL_ID = 'STT-v1'
MODEL_AUDIO_RATE = 16000

# ---------------------------- AUTOCORRECTOR CONSTANTS ---------------------------- #

MAX_ERROR = 3 
MAX_WORDS_TO_PREDICT = 3 

SANSKRIT_DICT_PATH = "Sanskrit Data/sanskrit_dictionary_ascii.json"


# ---------------------------- PREDICTOR CONSTANTS ---------------------------- #

CACHE_AUDIO_FILE_PATH = "Cache/audio.webm"