from os import listdir
from os.path import isfile,join
import numpy as np
import librosa
from librosa import feature
from glob import glob
import matplotlib.pyplot as plt
import librosa.display


dosya_1="audio_and_txt_filess"
ses_dosyalari=glob(dosya_1 + "/*.wav")


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
