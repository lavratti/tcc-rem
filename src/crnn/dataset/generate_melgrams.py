import os
import librosa
import matplotlib
import numpy as np
import pylab
import librosa.display

all_melgrams = []
all_melgrams_p = []

for n in range(1, 1001):

    print("Generating melgram for sample #{:04d}... ".format(n), end="")

    filename = "{}.mp3".format(n)
    path = os.path.join("clips_45seconds", filename)
    sig, fs = librosa.load(path)

    melgram = librosa.feature.melspectrogram(y=sig, sr=fs)
    melgram_p = librosa.power_to_db(melgram, ref=np.max)

    all_melgrams.append(melgram)
    all_melgrams_p.append(melgram_p)

    # Save linear
    with open("melgrams/melgram_linear/{}.csv".format(n), "wb") as f:
        np.savetxt(f, melgram, delimiter=",")
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(melgram)
    pylab.savefig("melgrams/melgram_linear/{}.png".format(n), bbox_inches=None, pad_inches=0)
    pylab.close()

    # Save Power to dB
    with open("melgrams/melgram_power_to_db/{}.csv".format(n), "wb") as f:
        np.savetxt(f, melgram_p, delimiter=",")
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(melgram_p)
    pylab.savefig("melgrams/melgram_power_to_db/{}.png".format(n), bbox_inches=None, pad_inches=0)
    pylab.close()

    print("Done! ({:3.1f}% completed)".format(n/10))