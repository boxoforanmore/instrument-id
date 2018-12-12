from pathlib import Path
import numpy as numpy
import scipy
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Two separate directories to guarantee that some from each label are used
# (vs RandomSplit)
INS_DIR = 'data/instruments'
TEST_DIR = 'data/test/instruments'


INSTRUMENTS = ['accordion', 'fiddle', 'flute', 'pennywhistle', 'uilleann']


def load_ceps(instruments, base_dir=INS_DIR):
    data = []
    labels = []

    for label, instrument in enumerate(instruments):
        ins_dir = Path(base_dir) / instrument

        for fn in ins_dir.glob('*.ceps.npy'):
            ceps = np.load(fn)
            num_ceps = len(ceps)

            # Average per coefficient over all frames for better generalization
            data.append(np.mean(ceps[num_ceps//2:(num_ceps*9)//10], axis=0))
            labels.append(label)

    return np.array(data), np.array(labels)

# Try for 70/30 split of data
X_train, y_train = load_ceps(instruments=INSTRUMENTS)
X_test, y_test = load_ceps(instruments=INSTRUMENTS, base_dir=TEST_DIR)

print()
print('X_train => Rows: %d, Columns: %d' % (X_train.shape[0], X_train.shape[1]))
print()
print('X_test  => Rows: %d, Columns: %d' % (X_test.shape[0], X_test.shape[1]))
print()
print()


# Set random seed
np.random.seed(123)
tf.set_random_seed(123)

# Onehot encode the labels
y_train_onehot = keras.utils.to_categorical(y_train)

print()
print('First 3 labels: ', y_train[:3])
print()
print('First 3 labels (one-hot): \n', y_train_onehot[:3])
print()


# Add a feedforward network
model = keras.models.Sequential()

# Input layer; input dimensions must match number of features in the training set
# Number of output and input units in two consecutive layers must also match

model.add(keras.layers.Dense(units=200, input_dim=X_train.shape[1],
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='tanh'))

model.add(keras.layers.Dense(units=200, input_dim=200,
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='tanh'))

model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=200, 
                             kernel_initializer='glorot_uniform',
                             bias_initializer='zeros',
                             activation='softmax'))

# Using SGD for better base performance and efficient activation; 
# need to play with decay rate
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
adadelta_optimizer = keras.optimizers.Adadelta()
rms_prop_optimizer = keras.optimizers.RMSprop()
nadam_optimizer = keras.optimizers.Nadam()

# Crossentropy is the generalization of logistic regression for
# multiclass predictions via softmax
model.compile(optimizer=rms_prop_optimizer, loss='categorical_crossentropy')


# Train with fit method
history = model.fit(X_train, y_train_onehot,
                    batch_size=40, epochs=75, verbose=1,
                    validation_split=0.1)


# Predict class labels (return class labels as integers)
y_train_pred = model.predict_classes(X_train, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]

print()
print('First 3 predictions: ', y_train_pred[:3])
print()
print('Training accuracy: %.2f%%' % (train_acc * 100))
print()

y_test_pred = model.predict_classes(X_test, verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print('Test accuracy: %.2f%%' % (test_acc * 100))
print()


model.save('model/instrument_id.h5', overwrite=True)
