import time
from pathlib import Path
import numpy as np
import scipy
import librosa

# The training and test directories
INS_DIR = 'data/instruments'
TEST_DIR = 'data/test/instruments'

INSTRUMENTS = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']


def generate_fft(instruments=INSTRUMENTS, base_dir=INS_DIR):
    data = []
    labels = []
    print("Number of Samples per Instrument in %s:" % (base_dir))
    
    for label, instrument in enumerate(instruments):
        ins_dir = Path(base_dir) / instrument
        num_labels = 0
        # Since I do know the file type of the files
        # I am simply globbing them all as mp3s
        for fname in ins_dir.glob(str(instrument)+'*.*'):
            X, sample_rate = librosa.core.load(fname)
            fft_features = abs(scipy.fft(X, sample_rate))
            
            data.append(fft_features)
            labels.append(label)
            
            num_labels += 1
        print('\t%i %s samples' % (num_labels, instrument))
    
    print()
    return np.array(data), np.array(labels)
         
total_time = 0

start_time = time.time()

X_train, y_train = generate_fft()
X_test, y_test = generate_fft(base_dir=TEST_DIR)

end_time = time.time()

print('X_train => Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('X_test  => Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))
total_time = preprocess_time = end_time-start_time
print('\nTime Taken: %s seconds' % (preprocess_time))
print('\n\n\n')


import tensorflow as tf
import tensorflow.keras as keras


# Get the one-hot encoded labels
y_train_onehot = keras.utils.to_categorical(y_train)
y_test_onehot = keras.utils.to_categorical(y_test)

print('First 3 labels:', y_train[:3])
print('First 3 labels (one-hot):\n', y_train_onehot[:3])
print('\n\n\n')


np.random.seed(4)
tf.set_random_seed(4)

start_time = time.time()

model = keras.models.Sequential()

model.add(keras.layers.Dense(units=100, input_dim=X_train.shape[1],
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='selu'))

model.add(keras.layers.Dense(units=100, input_dim=100,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='relu'))

model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=100,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-8, momentum=.9)
adadelta_optimizer = keras.optimizers.Adadelta()
rms_prop_optimizer = keras.optimizers.RMSprop()
nadam_optimizer = keras.optimizers.Nadam()

model.compile(optimizer=adadelta_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set verbose to 1 here to see a more detailed output
history = model.fit(X_train, y_train_onehot,
                    batch_size=100, epochs=50, verbose=0,
                    validation_split=0.2)


print('\n\n\n')
print('Training Data:')
training_scores = model.evaluate(X_train, y_train_onehot, verbose=0)
print('Model Loss\t:\t%.2f%%' % (training_scores[0]*100))
print('Model Accuracy\t:\t%.2f%%\n\n' % (training_scores[1]*100))

print('Test Data')
test_scores = model.evaluate(X_test, y_test_onehot, verbose=0)
print('Model Loss\t:\t%.2f%%' % (test_scores[0]*100))
print('Model Accuracy\t:\t%.2f%%' % (test_scores[1]*100))

end_time = time.time()
training_time = end_time - start_time

# Update total time taken in preprocessing
total_time += training_time

print('\nTime Taken: %s seconds' % (training_time))
print('\n\n\n')

import matplotlib.pyplot as plt

train_accuracy = np.array(history.history['acc'])*100
test_accuracy = np.array(history.history['val_acc'])*100
train_loss = np.array(history.history['loss'])*100
test_loss = np.array(history.history['val_loss'])*100

plt.figure(figsize=(22,14))
plt.plot(train_accuracy)
plt.plot(test_accuracy)

plt.title('FFT Accuracy/Loss (%) vs Epoch')
plt.plot()
plt.yticks(list(range(0, 1700, 50)))
plt.xticks(list(range(0, 50, 2)))
plt.plot(train_loss)
plt.plot(test_loss)
plt.ylabel('Accuracy/Loss (%)')
plt.xlabel('Epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='best')
plt.show()


y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score, classification_report

ac = accuracy_score(y_train, y_train_pred)
report = classification_report(y_train, y_train_pred)

print("Train Report: Accuracy => %.2f%%\n" % (ac*100))
print(report, '\n\n')


ac = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)

print("Test Report: Accuracy => %.2f%%\n" % (ac*100))
print(report)



import matplotlib.pyplot as plt

sample_indices = [np.random.randint(0, 39), np.random.randint(40, 79), 
                  np.random.randint(80, 119), np.random.randint(120, 159),
                  np.random.randint(160, 199), np.random.randint(200, 239),
                  np.random.randint(240, 279), np.random.randint(280, 319),
                  np.random.randint(320, 359), np.random.randint(360, 399)]

indices = np.array(range(len(X_train[0]+1)//2))
ticks = list(range(0, len(X_train[0])//2,1000))
alt = False

for index in sample_indices:
    if alt == False:
        plt.figure(figsize=(20,6))
        plt.subplot(121)
        plt.xticks(ticks)
    else:
        plt.subplot(122)
        plt.xticks(ticks)
    plt.plot(indices, X_train[index][0:len(X_train[index]+1)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT for ' + INSTRUMENTS[y_train[index]] + ' at index ' + str(index))
    if alt == False:
        alt = True
    else:
        alt = False
        plt.show()




sample_ranges = [(0, 79),(79, 159), (160, 239), (240, 319), (320, 399)]


indices = np.array(range(len(X_train[index]+1)//2))

alt = False

for index, num_range in enumerate(sample_ranges):
    plt.figure(figsize=(22,8))
    for graph in range(num_range[0], num_range[1]):
        plt.plot(indices, X_train[graph][0:len(X_train[graph]+1)//2])
    plt.title('Layered FFT: ' + INSTRUMENTS[index])
    plt.xticks(ticks)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()



