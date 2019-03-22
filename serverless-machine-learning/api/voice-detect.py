
print('start')

import numpy as np              # computing module
from IPython.display import Audio  # play the audio
import librosa                  # Audio management
import pickle

target_file = './sample_data/2019-03-17T14_30_44.622Z.wav'

print(target_file)
data, sr = librosa.load(target_file)
Audio(data, rate = sr)

print(len(data))
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

#scaler = StandardScaler() # initialise the scaler
scaler = pickle.load(open('knn_voice_detect_scaler.sav', 'rb'))
#test_data = scaler.fit_transform([df])
transformed_data = scaler.transform([[spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc])
print(transformed_data)

filename = 'knn_voice_detect_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.predict(np.array([[spec_cent, zcr, chroma_stft, rolloff, spec_bw] + mfcc]))

result = loaded_model.predict(transformed_data)

map_to_labels = {0: 'MA', 1: 'FE'}

print(result)
print('result :',map_to_labels[result[0]])