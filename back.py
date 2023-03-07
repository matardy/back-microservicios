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
    
    return encoder.inverse_transform(y_pred)


    

@app.route('/backend/audio', methods=['POST'])
def receive_audio():
    audio_file = request.files['audio']
    audio = AudioSegment.from_file(audio_file)
    
    # Set the path and filename for the downloaded audio file
    filename, extension = audio_file.filename.split('.')
    filepath = 'audio/' + f'{filename}_downloaded.{extension}'
    
    # Write the audio file data to the downloaded file
    audio.export(filepath, format=extension)
    
    # load data
    rate, data = wavfile.read(filepath)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write("audio/mywav_reduced_noise.wav", rate, reduced_noise)
    
    predict_function(filepath)
    # Aquí puedes trabajar con el archivo de audio recibido, por ejemplo:
    # guardar el archivo en el disco duro
    # procesar el archivo de audio
    # devolver una respuesta al cliente, etc.
    response = jsonify({'message': predict_function(filepath)})
    response.hearders.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # Por ejemplo, aquí devolvemos una respuesta simple indicando que se recibió el archivo de audio
    return response


if __name__ == '__main__':
    app.run(debug=True, port=3000)
