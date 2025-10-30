Speech Emotion Recognition
Overview

This project focuses on building a deep learning model capable of recognizing human emotions from speech signals. The goal is to analyze audio recordings, extract meaningful acoustic features, and classify them into emotional categories such as happy, sad, angry, and neutral. Speech emotion recognition (SER) plays a vital role in improving human-computer interaction, virtual assistants, and sentiment-aware applications.

Problem Statement

Understanding human emotions from speech is a challenging task due to variations in tone, pitch, and speaking style. The aim of this project is to develop an efficient model that accurately classifies emotions using audio feature analysis and deep learning techniques, specifically Long Short-Term Memory (LSTM) networks.

Dataset

The project uses publicly available emotional speech datasets such as:

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

TESS (Toronto Emotional Speech Set)

SAVEE (Surrey Audio-Visual Expressed Emotion)

Each dataset includes multiple speakers expressing different emotions through speech.

Key Attributes:

Audio recordings sampled at 44.1 kHz

Emotions: Neutral, Happy, Sad, Angry, Fearful, Disgust, and Surprise

Format: WAV files

Approach
1. Feature Extraction

Extracted key features from audio signals using Librosa:

MFCC (Mel-Frequency Cepstral Coefficients)

Chroma Features

Spectral Centroid

Zero Crossing Rate

Spectral Roll-off

These features capture essential frequency and temporal characteristics related to emotional cues in speech.

2. Data Preprocessing

Converted audio files into numerical feature arrays.

Normalized feature data for consistent scaling.

Split dataset into training, validation, and test sets.

3. Model Building

Implemented an LSTM-based neural network using TensorFlow and Keras to model temporal dependencies in speech features.
Key layers include:

LSTM layers for sequential learning

Dropout layers for regularization

Dense layers for classification

4. Model Training and Evaluation

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Confusion Matrix, and Classification Report

Achieved a validation accuracy of approximately 84%.

5. Visualization

Plotted waveform and spectrogram representations for emotion samples.

Visualized feature distributions and confusion matrix for performance interpretation.

Technologies Used

Programming Language: Python

Libraries: Librosa, NumPy, Pandas, Matplotlib, TensorFlow, Keras, Scikit-learn

Environment: Jupyter Notebook

Results

The LSTM model achieved strong performance in recognizing emotions from speech.

MFCC and Chroma features proved most influential for accurate emotion classification.

The model demonstrates robustness across multiple speakers and datasets.

Conclusion

The project successfully demonstrates the application of deep learning in emotion recognition from speech. By analyzing frequency-based and temporal features, the LSTM model provides valuable insights into emotional expressions. This work can be extended to applications such as emotion-aware virtual assistants, mental health monitoring, and affective computing.

Future Work

Experiment with CNN-LSTM hybrid models for improved accuracy.

Extend to multilingual emotion recognition.

Deploy as a web-based or real-time application using Flask or Streamlit.

How to Run

Clone this repository:

git clone https://github.com/your-username/speech-emotion-recognition.git


Navigate to the project directory:

cd speech-emotion-recognition


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook
