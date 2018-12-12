import tensorflow as tf
import tensorflow.keras as keras
import scipy
import numpy
import librosa
from python_speech_features import mfcc
from sys import argv
from pathlib import Path

INSTRUMENTS = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']

def generate_ceps(fname):
    X, sample_rate = librosa.core.load(fname)

    ceps = mfcc(X)

    return [np.mean(ceps[num_ceps//2:(num_ceps*9)//10], axis=0)]


if '__main__' == __name__:
    length = len(argv)
    model = keras.models.load_model('model/instrument_id.h5')


    fnum = 1
    while (fnum < length):
        print()
        print('------------------------------------')
        print()
        print('Converting file: ', str(argv[fnum]))
        ceps = generate_ceps(fnum)

        print()
        print('Classifying sample with model')
        test_pred = model.predict_classes(ceps, verbose=0)
        print('Guess: %s has an %s' % (fname, INSTRUMENTS[test_pred]))
        print()
