from pathlib import Path
from python_speech_features import mfcc
import numpy as np
import scipy
import librosa

# This script should generate and save ffts as a preprocessing step

# The training and test data directories
INS_DIR = 'data/instruments'
TEST_DIR = 'data/test/instruments'

def generate_ceps(fname):
    # Librosa is used here as it can load mp3s
    # --> Scipy can only load wavs and another 
    #     decompression could lead to some 
    #     corruption of data
    X, sample_rate = librosa.core.load(fname)

    # Save the serialized fft for use later
    np.save(Path(fname).with_suffix('.ceps'), mfcc(X))


# Generate ceps for the training data
print('Processing training data...')
for mp3_file in Path(INS_DIR).glob('**/*.mp3'):
    generate_ceps(mp3_file)


print('Processing test data...')
# Generate ceps for the test data
for mp3_file in Path(TEST_DIR).glob('**/*.mp3'):
    generate_ceps(mp3_file)
