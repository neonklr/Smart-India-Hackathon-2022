# ---------------------------- MODEL RUNNER CONSTANTS ---------------------------- #

MODEL_PATHS = {
    'STT-online-v1': 'addy88/wav2vec2-sanskrit-stt',
    'Quant-heroku': '/app/Models/vakyansh-wav2vec2-sanskrit-sam-60-quantized/vakyansh-wav2vec2-sanskrit-sam-60_quant.pt',

    'STT-v1': 'Models/wav2vec2-sanskrit-stt',
    'STT-v2': 'Models/vakyansh-wav2vec2-sanskrit-sam-60', # TODO: Error Prone? (build error catcher)

    'Quant-v2': 'Models/vakyansh-wav2vec2-sanskrit-sam-60-quantized/vakyansh-wav2vec2-sanskrit-sam-60_quant.pt'
}

DEFAULT_MODEL_ID = 'Quant-v2'
MODEL_AUDIO_RATE = 16000

# ---------------------------- AUTOCORRECTOR CONSTANTS ---------------------------- #

MAX_ERROR = 3
MAX_WORDS_TO_PREDICT = 3

SANSKRIT_DICT_PATH = "Sanskrit Data/sanskrit_dictionary_ascii.json"


# ---------------------------- PREDICTOR CONSTANTS ---------------------------- #

CACHE_AUDIO_FILE_PATH = "Cache/audio.webm"



QUANTIZED_MODEL_HEADER = "Quant"
BINARY_MODEL_HEADER = "STT"