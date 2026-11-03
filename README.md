# No-Mimic

**Author:** Sudip Patra  
**Instructor:** Prof. Anirban Mukhopadhyay, University of Kalyani

No-Mimic is a voice authenticity detection project that classifies whether an audio clip sounds like a real human voice or an AI-generated synthetic voice. The system combines audio preprocessing, handcrafted feature extraction, and a deep learning model based on a CNN-BiLSTM architecture.

It includes a Streamlit web app for uploading audio files and reviewing predictions with visual explanations.

## Features

- Real vs AI voice classification for uploaded audio
- Audio preprocessing and normalization
- Feature extraction using Mel spectrograms and MFCCs
- CNN + BiLSTM model for temporal voice pattern analysis
- Streamlit dashboard for inference and result visualization
- Training pipeline with validation and early stopping
- Synthetic audio generation support for dataset expansion

## Project Structure

```text
No-Mimic/
├── config.py                  # Global project configuration
├── requirements.txt           # Python dependencies
├── streamlit_app.py           # Streamlit web application
├── data/
│   ├── raw/                  # Raw audio input folders
│   │   ├── authentic/
│   │   └── fake/
│   ├── processed/            # Moved processed audio files
│   ├── features/            # Extracted .pt feature files
│   └── feature_params.pt    # Saved feature parameters
├── models/
│   ├── best_model.pth       # Best trained checkpoint
│   └── training_history.json
├── scripts/
│   └── prepare_dataset.py   # Dataset preparation and feature extraction
├── src/
│   ├── dataset.py            # Data loading and splits
│   ├── evaluate.py           # Evaluation pipeline
│   ├── feature_extractor.py  # Audio feature extraction
│   ├── model.py              # Voice authenticity model
│   ├── predict.py            # Prediction/inference logic
│   ├── preprocessor.py       # Audio loading and cleaning
│   └── train.py              # Model training script
└── README.md
```

## Tech Stack

- Python
- PyTorch
- Librosa
- NumPy / Pandas / SciPy
- scikit-learn
- Streamlit
- Matplotlib / Plotly
- gTTS, pyttsx3, edge-tts (for synthetic data generation)

## Dataset Setup

The project expects raw audio data under these directories:

```text
data/raw/authentic/
data/raw/fake/
```

Place authentic human voice recordings in `data/raw/authentic/` and synthetic or AI-generated recordings in `data/raw/fake/`.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd No-Mimic
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Preparing the Dataset

Generate processed feature files from the raw audio data:

```bash
python scripts/prepare_dataset.py
```

This script:

- loads audio files,
- trims and normalizes them,
- splits long clips into fixed-length segments,
- extracts features,
- saves them as `.pt` files in `data/features/`.

## Training the Model

Run the training script:

```bash
python src/train.py
```

The script trains the model using the prepared dataset, validates it during training, and saves the best checkpoint to:

```text
models/best_model.pth
```

## Running the Web App

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown by Streamlit in the browser and upload an audio file for classification.

## Prediction Workflow

The app pipeline is:

1. Upload audio file
2. Preprocess waveform
3. Extract features
4. Run model inference
5. Show prediction, confidence, and explanation reasons

## Model Notes

The architecture used in this project is a CNN-BiLSTM hybrid model:

- CNN blocks capture spectral patterns from Mel spectrograms
- BiLSTM learns temporal dependencies in sequential voice characteristics
- Self-attention pooling emphasizes the most informative time steps
- Final classifier predicts whether the input is authentic or AI-generated

## Example Commands

```bash
# Prepare features
python scripts/prepare_dataset.py

# Train model
python src/train.py

# Launch app
streamlit run streamlit_app.py
```

## Notes

- The dataset will be added to the repository soon.
- The model checkpoint is stored in `models/best_model.pth` and is used by the prediction pipeline automatically.
- If the model file is missing, prediction may fall back to untrained/random weights.
- Audio inputs should be clean and sufficiently long enough for the 3-second segmentation pipeline to work correctly.


