from flask import Flask
from flask import request
from flask_cors import CORS
import boto3
import base64
from datetime import datetime
import os

import numpy as np              # computing module
from IPython.display import Audio  # play the audio
import librosa                  # Audio management
import pickle

#import voice_detect

BUCKET_NAME = 'machine-learning-python'
MODEL_FILE_NAME = 'voice-detect/knn_voice_detect_model.pkl'
SCALER_FILE_NAME = 'voice-detect/knn_voice_detect_scaler.pkl'

app = Flask(__name__)
CORS(app)

S3 = boto3.client('s3', region_name='ap-northeast-1')


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@app.route('/', methods=['POST'])
def index():
    voice_raw_data = request.form['voice-data']
    voice_data = base64.decodebytes(voice_raw_data.encode())
    file_name = datetime.now().strftime('%s') + '.wav'
    print(file_name)
    file = open(file_name, 'wb')
    file.write(voice_data)
    file.close()

    # result = voice_detect.detect_sex_by_sound(file_name)
    result = detect_sex_by_sound(file_name)
    os.remove(file_name)
    return result


@app.route('/', methods=['GET'])
def get():
    return "success!"

@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()

    model = pickle.loads(model_str)

    return model


def predict(data):
    model = load_model(MODEL_FILE_NAME)

    return model.predict(data).tolist()

def detect_sex_by_sound(target_file):
    print(target_file)
    data, sr = librosa.load(target_file)
    Audio(data, rate = sr)
    middle = len(data) // 2
    data = data[middle - sr // 2:middle + sr // 2]

    #### Spectral Centroid extraction ######
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))
    ##### Zero Crossing Rate extraction ######
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data))
    ##### Chroma Frequencies extraction ######
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sr))
    ##### Spectral Roll-off extraction ######
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr))
    ##### Spectral Bandwidth extraction ######
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sr))
    ##### Mel-frequency cepstral coefficients (MFCC) extraction ######
    mfcc = [np.mean(x) for x in librosa.feature.mfcc(y=data, sr=sr)]

    # scaler = pickle.load(open('knn_voice_detect_scaler.pkl', 'rb'))
    scaler = load_model(SCALER_FILE_NAME)
    transformed_data = scaler.transform([[spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc])
    print(transformed_data)

    # loaded_model = pickle.load(open('knn_voice_detect_model.pkl', 'rb'))
    loaded_model = load_model(MODEL_FILE_NAME)
    result = loaded_model.predict(transformed_data)
    map_to_labels = {0: 'MA', 1: 'FE'}

    print(result)
    print('result :',map_to_labels[result[0]])

    return map_to_labels[result[0]]


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')
