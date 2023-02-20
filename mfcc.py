from os import listdir
from os.path import isfile,join
import numpy as np
import librosa
from librosa import feature
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import csv

dosya_1="audio_and_txt_filess"
ses_dosyalari=glob(dosya_1 + "/*.wav")
print(ses_dosyalari[712])

for i in range(len(ses_dosyalari)):
    if "179_1b1_Tc_sc_Meditron_sr200_time_shifted.wav" in ses_dosyalari[i]:
        print(i)

filenames=[f for f in listdir(dosya_1) if (isfile(join(dosya_1,f)) and f.endswith(".wav"))]


p_id_in_file=[]
for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file= np.array(p_id_in_file)
## mfcc features ekleyim
##makine öğrenmesiyle daha etkili sonuç aalbilirim, mö+dö


def get_spectrogram(file):

    y, sr = librosa.load(file, sr=4000)
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    s_db = librosa.power_to_db(s, ref=np.max)
    fig = plt.figure(figsize=(12, 8))
    img = librosa.display.specshow(s_db, sr=sr)
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    plt.axis("off")

    return plt.savefig(f"{file}.png", bbox_inches="tight", pad_inches=0)

for file in ses_dosyalari:

    get_spectrogram(file)
