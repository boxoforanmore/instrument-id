# Classifier that uses ffts to classify instruments used in Irish Traditional Music #

=============

## Current Model: ##
* Saved as an h5 file in the `models` directory
* Uses data processed from mfcc (not fft)
* ~ 88% test accuracy, ~97% training accuracy
	* Overfits the training data a fair amount, but this could be slightly fixed by utilizing a larger dataset
* Only classifies on accordion, fiddle, flute, pennywhistle, and uilleann pipes
* Hyperparameters were hand-tuned to some degree, Grid Search (or a similar optimization finder)  will be run in the future
* Will most likely switch to a CNN or an RNN
* Needs a script to load, preprocess, and classify a new data


=============


## File Notes ##
### `con2wav.sh` ###
* Uses sox to convert mp3s to wav


### `trimFiles.sh` ###
* Uses sox to trim mp3 files down to 30 second increments


### `gen_fft.py` and `gen_ceps.py` ###
* Processes files with FFT or ceps


### `train_fft.py` ###
* Trains a keras model on the training and test data
* Current test data accuracy ~ 62%


### `train_ceps.py` ###
* Trains and saves a keras model on the training and test data



