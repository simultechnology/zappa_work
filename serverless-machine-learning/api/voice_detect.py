
import numpy as np              # computing module
from IPython.display import Audio  # play the audio
import librosa                  # Audio management
import pickle

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

    scaler = pickle.load(open('knn_voice_detect_scaler.pkl', 'rb'))
    transformed_data = scaler.transform([[spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc])
    print(transformed_data)

    loaded_model = pickle.load(open('knn_voice_detect_model.pkl', 'rb'))
    result = loaded_model.predict(transformed_data)
    map_to_labels = {0: 'MA', 1: 'FE'}

    print(result)
    print('result :',map_to_labels[result[0]])

    return map_to_labels[result[0]]

