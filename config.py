import os

class config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    AUTHENTIC_DIR = os.path.join(RAW_DIR, "authentic")
    FAKE_DIR = os.path.join(RAW_DIR, "fake")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    FEATURES_DIR = os.path.join(DATA_DIR, "features")
    FEATURES_PT_PATH = FEATURES_DIR  # Alias used by dataset.py for loading individual .pt files
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth") 
    
    # Audio Preprocessing
    SAMPLE_RATE = 16000
    DURATION = 3
    NUM_SAMPLES = SAMPLE_RATE * DURATION
    SILENCE_THRESHOLD_DB = -30
    
    # Feature Extraction
    N_MELS = 80
    N_MFCC = 40
    N_FFT = 1024
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    F_MIN = 20
    F_MAX = 8000
    NUM_TIME_FRAMES = NUM_SAMPLES // HOP_LENGTH
    
    # Model Architecture
    CNN_CHANNELS = [2, 32, 64, 128]
    CNN_KERNEL_SIZE = 3
    CNN_PADDING = 1
    CNN_DROPOUT = 0.3
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.3
    FC_HIDDEN_SIZE = 64
    FC_DROPOUT = 0.4
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 11
    TRAIN_SPLIT = 0.80
    VAL_SPLIT = 0.10
    TEST_SPLIT = 0.10
    RANDOM_SEED = 42
    
    # Data Augmentation
    AUGMENT_NOISE_PROB = 0.3
    AUGMENT_NOISE_FACTOR = 0.005
    AUGMENT_TIME_SHIFT_PROB = 0.3
    AUGMENT_TIME_SHIFT_MAX = 1600
    
    # Streamlit
    STREAMLIT_PORT = 8501
    PAGE_TITLE = "No-Mimic"
    PAGE_ICON = None  # Uses generated favicon.png
    MAX_UPLOAD_SIZE_MB = 50
    ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "webm", "mp4"}
    
    # Dataset Generation
    NUM_SYNTHETIC_SAMPLES = 2000
    NUM_AUTHENTIC_SAMPLES = 2000
    SYNTHETIC_LANGUAGES = ["en"]
    SYNTHETIC_TTS_ENGINES = ["gtts", "edge-tts", "pyttsx3"]
    SYNTHETIC_SPEED_RANGE = (0.8, 1.3)
    SYNTHETIC_NOISE_PROB = 0.3
    SYNTHETIC_NOISE_SNR_DB = (15, 30)
