import tensorflow as tf
import tensorflow.keras as keras
import scipy
import numpy as np
import librosa
from python_speech_features import mfcc
from sys import argv
from pathlib import Path

INSTRUMENTS = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']

def generate_ceps(fname):
    data = []
    X, sample_rate = librosa.core.load(fname)

    ceps = mfcc(X)
    num_ceps = len(ceps)

    return np.mean(ceps, axis=0)


if '__main__' == __name__:
    length = len(argv)
    model = keras.models.load_model('model/instrument_id.h5')

    fnum = 1
    while (fnum < length):
        print()
        print('------------------------------------')
        print()
        print('Converting file: ', str(argv[fnum]))
        ceps = generate_ceps(argv[fnum])
        data = []
        print(ceps.shape)
        data.append(ceps)
        ceps = np.array(data)
        print()
        print('Classifying sample with model')
        test_pred = model.predict_classes(ceps, verbose=0)
        print('Guess: %s has an %s' % (argv[fnum], INSTRUMENTS[test_pred[0]]))
        print()
        fnum += 1
