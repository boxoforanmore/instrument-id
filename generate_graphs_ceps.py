from python_speech_features import mfcc
import time
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

INSTRUMENTS = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']
INS_DIR = 'data/instruments'
TEST_DIR = 'data/test/instruments'

np.random.seed(4)
tf.set_random_seed(4)

# Storing the old fft data for later comparisons
#fft_X_train, fft_y_train = X_train, y_train
#fft_X_test, fft_y_test = X_test, y_test
#fft_time =int(total_time)
total_time = 0

def generate_ceps(instruments=INSTRUMENTS, base_dir=INS_DIR):
    data = []
    labels = []
    print("Number of Samples per Instrument in %s:" % (base_dir))
    
    for label, instrument in enumerate(instruments):
        ins_dir = Path(base_dir) / instrument
        num_labels = 0

        for fname in ins_dir.glob(str(instrument)+'*.*'):
            X, sample_rate = librosa.core.load(fname)
            ceps_features = mfcc(X)
            num_ceps = len(ceps_features)
            
            # Average per coefficient over all frames for better generalization and
            # better noise reduction
            # data.append(np.mean(ceps_features[num_ceps//2:(num_ceps*9)//10], axis=0))
            data.append(np.mean(ceps_features, axis=0))
            labels.append(label)
            num_labels += 1
            
        print('\t%i %s samples' % (num_labels, instrument))
    
    print()
    return np.array(data), np.array(labels)


start_time = time.time()

X_train, y_train = generate_ceps()
X_test, y_test = generate_ceps(base_dir=TEST_DIR)

end_time = time.time()


print('X_train => Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('X_test  => Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))
preprocess_time = end_time-start_time
total_time += preprocess_time
print('\nTime Taken: %s seconds' % (preprocess_time))



# Get the one-hot encoded labels
y_train_onehot = keras.utils.to_categorical(y_train)
y_test_onehot = keras.utils.to_categorical(y_test)
print('First 3 labels:', y_train[:3])
print('First 3 labels (one-hot):\n', y_train_onehot[:3])



np.random.seed(4)
tf.set_random_seed(4)

model = keras.models.Sequential()

model.add(keras.layers.Dense(units=200, input_dim=X_train.shape[1],
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='selu'))

model.add(keras.layers.Dense(units=120, input_dim=200,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='relu'))

model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=120,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
adadelta_optimizer = keras.optimizers.Adadelta()
rms_prop_optimizer = keras.optimizers.RMSprop(lr=0.001, decay=1e-8)
nadam_optimizer = keras.optimizers.Nadam()

model.compile(optimizer=rms_prop_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

# Set verbose to 1 here to see a more detailed output
history = model.fit(X_train, y_train_onehot,
                    batch_size=200, epochs=66, verbose=0,
                    validation_split=0.1)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

first_model = model
first_history = history

print('Training Data:')
training_scores = model.evaluate(X_train, y_train_onehot, verbose=0)
print('Model Loss\t:\t%.2f%%' % (training_scores[0]*100))
print('Model Accuracy\t:\t%.2f%%\n\n' % (training_scores[1]*100))

print('Test Data')
test_scores = model.evaluate(X_test, y_test_onehot, verbose=0)
print('Model Loss\t:\t%.2f%%' % (test_scores[0]*100))
print('Model Accuracy\t:\t%.2f%%' % (test_scores[1]*100))


end_time = time.time()
ceps_time = end_time - start_time
total_time += ceps_time
print('\nTime Taken: %s seconds' % (ceps_time))


train_accuracy = np.array(history.history['acc'])*100
test_accuracy = np.array(history.history['val_acc'])*100
train_loss = np.array(history.history['loss'])*100
test_loss = np.array(history.history['val_loss'])*100

plt.figure(figsize=(22,14))
plt.plot(train_accuracy)
plt.plot(test_accuracy)


plt.plot()
plt.yticks(list(range(0, 400, 20)))
plt.xticks(list(range(0, 76, 2)))
plt.plot(train_loss)
plt.plot(test_loss)
plt.ylabel('Accuracy/Loss (%)')
plt.xlabel('Epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='best')
plt.show()



print('\n\n\n')

from sklearn.metrics import accuracy_score, classification_report

ac = accuracy_score(y_train, y_train_pred)
report = classification_report(y_train, y_train_pred)

print("Train Report: Accuracy => %.2f%%\n" % (ac*100))
print(report, '\n\n')


ac = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)

print("Test Report: Accuracy => %.2f%%\n" % (ac*100))
print(report)
print('\n\n\n')



sample_ranges = [(0, 79),(79, 159), (160, 239), (240, 319), (320, 399)]

indices = np.array(range(len(X_train[0])))

for index, num_range in enumerate(sample_ranges):
    plt.figure(figsize=(12,4))
    for graph in range(num_range[0], num_range[1]):
        plt.plot(indices, X_train[graph])
        plt.xticks(indices)
    plt.title('Layered MFCC: ' + INSTRUMENTS[index])
    plt.show()


sample_indices = [np.random.randint(0, 39), np.random.randint(40, 79), 
                  np.random.randint(80, 119), np.random.randint(120, 159),
                  np.random.randint(160, 199), np.random.randint(200, 239),
                  np.random.randint(240, 279), np.random.randint(280, 319),
                  np.random.randint(320, 359), np.random.randint(360, 399)]


indices = np.array(range(len(X_train[0])))
ticks = list(range(0, len(X_train[0])))
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
    plt.title(INSTRUMENTS[y_train[index]] + ' at index ' + str(index))
    if alt == False:
        alt = True
    else:
        alt = False
        plt.show()
