from os import listdir
from os.path import isfile,join
import numpy as np
import librosa
from librosa import feature
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

dosya_1="audio_and_txt_filess"
ses_dosyalari=glob(dosya_1 + "/*.wav")

filenames=[f for f in listdir(dosya_1) if (isfile(join(dosya_1,f)) and f.endswith(".wav"))]

p_id_in_file=[]
for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file= np.array(p_id_in_file)


fn_list_i = [
        feature.chroma_stft,
        feature.spectral_centroid,
        feature.spectral_bandwidth,
        feature.spectral_rolloff,

    ]

fn_list_ii = [

        feature.zero_crossing_rate

    ]
def get_feature_vector(y, sr):

    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]

    feature_vector = feat_vect_i + feat_vect_ii
    feature_vector = np.round(feature_vector, 2)
    return feature_vector

filepaths=[join(dosya_1,f) for f in filenames]

tani_dosyasi=pd.read_csv("patient_diagnosis1.csv",header=None)
labels = np.array([tani_dosyasi[tani_dosyasi[0] == x][1].values[0] for x in p_id_in_file])

# build the matrix with normal audios featurized
audios_feat = []
for file in ses_dosyalari:
    '''
    y is the time series array of the audio file, a 1D np.ndarray
    sr is the sampling rate, a number
    '''
    y, sr = librosa.load(file, sr=4000)
    feature_vector = get_feature_vector(y, sr)
    #feature_vector=np.round(feature_vector,2)
    audios_feat.append(feature_vector)

print("finished feature extr. from: ", len(audios_feat))

audios_feat=np.array(audios_feat)
flat_features = [f.flatten() for f in audios_feat]

# Create a dataframe with the audio features and the ID
df = pd.DataFrame({'ID': p_id_in_file, 'Features': flat_features})

# Write the dataframe to a CSV file
df.to_csv("audio_and_text.csv", index=False)

# Read the audio features file
audio_features = pd.read_csv("audio_and_text.csv")

# Read the diagnosis file
diagnosis = pd.read_csv("patient_diagnosis1.csv",header=None)

# Rename the column of diagnosis
diagnosis = diagnosis.rename(columns={0: "ID", 1: "diagnosis"})

# Merge the two dataframes on the ID column
merged_data = pd.merge(audio_features, diagnosis, on='ID')

# Write the merged data to a new CSV file
merged_data.to_csv("augmented_data.csv", index=False)







