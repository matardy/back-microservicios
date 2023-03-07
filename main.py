from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import io
import soundfile as sf
import subprocess
from pydub import AudioSegment, effects
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

import librosa
import librosa.display
from scipy.io import wavfile
import noisereduce as nr
import soundfile
import joblib # para cargar el scaler.pkl
import pickle
import traceback




app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas de la aplicación

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


def predict_function(filename):
    scaler = joblib.load("model/scaler.pkl")
    with open('model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    # Load the trained model from file
    with open('model/svc_final_model.pkl', 'rb') as f:
        svc = pickle.load(f)

    audio_featured = extract_feature(filename, mfcc=True, chroma=True, mel=True)

    
    scaled_audio = scaler.transform([audio_featured])

    y_pred = svc.predict(scaled_audio)
    
    return str(encoder.inverse_transform(y_pred)[0])


    

@app.route('/backend/audio', methods=['POST'])
def receive_audio():
    try: 
        audio_file = request.files['audio']
        audio = AudioSegment.from_file(audio_file)
        
        # # Set the path and filename for the downloaded audio file
        filename, extension = audio_file.filename.split('.')
        filepath = 'audio/' + f'{filename}_downloaded.{extension}'
        
        # # Write the audio file data to the downloaded file
        audio.export(filepath, format=extension)
        
        
        
    
        response = jsonify({'message': predict_function(filepath)})
    except Exception as e:
        error_message = traceback.format_exc() 
        response = jsonify({'message': error_message}), 500 

    # response.hearders.add('Access-Control-Allow-Origin', '*')
    # response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    # response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # Por ejemplo, aquí devolvemos una respuesta simple indicando que se recibió el archivo de audio
    return response


if __name__ == '__main__':
    app.run(True, port=5000)
