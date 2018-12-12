from pathlib import Path
import numpy as np
import scipy
import librosa

# This script should generate and save ffts as a preprocessing step

# The training and test data directories
INS_DIR = 'data/instruments'
TEST_DIR = 'data/test/instruments'

def generate_fft(fname):
    # Librosa is used here as it can load mp3s
    # --> Scipy can only load wavs and another 
    #     decompression could lead to some 
    #     corruption of data
    X, sample_rate = librosa.core.load(fname)

    # Calculate the fft--scipy seems to give better
    # results here (the sample rate is given for 
    # overall consistency)
    fft_features = abs(scipy.fft(X[:4000], sample_rate))

    # Save the serialized fft for use later
    np.save(Path(fname).with_suffix('.fft'), fft_features)


# Generate fft for the training data
for mp3_file in Path(INS_DIR).glob('**/*.mp3'):
    generate_fft(mp3_file)


# Generate fft for the test data
for mp3_file in Path(TEST_DIR).glob('**/*.mp3'):
    generate_fft(mp3_file)
