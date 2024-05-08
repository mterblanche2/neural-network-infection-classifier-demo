# RUN ON LAPTOP
dir = 'D:/dev/teaching_repo/'
nrows = 2000000

###############################################################################################

import os
from datetime import datetime as dt
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
# IMPORT PACKAGES & MODULES
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def label_encode_targets(df):
    scaler = LabelEncoder().fit(df)
    labelled_target = to_categorical(scaler.transform(df))
    print(labelled_target.shape)
    return scaler, labelled_target


# SHOW TRAINING PLOTS

def plot_losses(history):
    loss = history.history['loss'][2:]
    val_loss = history.history['val_loss'][2:]
    loss_epochs = range(1, len(loss) + 1)
    plt.clf()
    plt.plot(loss_epochs, loss, 'bx', label='Training loss')
    plt.plot(loss_epochs, val_loss, 'ro', label='Validation loss')
    plt.title('Model training & validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    trainplot = plt.gcf()
    plt.show()
    trainplot.savefig(NAME + '--TRAINING PLOT.png', dpi=700)


def plot_accuracy(history):
    plt.clf()
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'r.', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Model training & validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    trainplot_acc = plt.gcf()
    plt.show()
    trainplot_acc.savefig(NAME + '--ACC.png', dpi=700)


def build_small_model(neurons):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[2], kernel_initializer='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])
    model.summary()
    return model


def build_big_model(neurons):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[2], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[4], kernel_initializer='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])
    model.summary()
    return model


def build_medium_model(neurons):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[2], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])
    model.summary()
    return model


# LOAD & PREPARE TRAIN DATA
condition = 'TRAIN.csv'
os.chdir(dir)
model = 'INFECTION'
date = dt.now().strftime('-%Y.%m.%d')
NAME = model + date
print(NAME)

# LOAD RAW DATA FILES
train_data = pd.read_csv(dir + 'TRAIN.CSV', header=0, nrows=200000)
print(train_data)

test_data = pd.read_csv(dir + 'TEST.CSV', header=0, nrows=50000)
print(test_data)

# GENERATE INPUT FEATURES & LABELS
X_train = train_data.iloc[:, 0:-1].values.astype('float32')
X_test = test_data.iloc[:, 0:-1].values.astype('float32')
print(X_train)

y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]
print(y_train)

scaler, labelled_y_train = label_encode_targets(y_train)
scaler, labelled_y_test = label_encode_targets(y_test)
print(labelled_y_train)

print('SCALER CLASSES:', scaler.classes_)

# INPUT & OUTPUT SHAPES
INPUT_SHAPE = X_train.shape[1]
OUTPUT_SHAPE = labelled_y_train.shape[1]

print(INPUT_SHAPE)
print(OUTPUT_SHAPE)

# SMALL MODEL: SHORT & SHALLOW

# neurons = [2, 2, 6]
#
# model = build_small_model(neurons)

# BIG MODEL: WIDE & DEEP

neurons = [256, 256, 256, 256, 6]
#
model = build_big_model(neurons)
#
# # SOMEHWERE IN-BETWEEN
#
# neurons = [256, 128, 64, 6]
#
# model = build_medium_model(neurons)

history = History()
filepath = NAME + '--best.weights.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy',
                                             verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto',
                                             save_freq="epoch")

stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=200, mode='auto')
csv_logger = CSVLogger(NAME + '_training-log.log', separator=',', append=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              mode='auto', cooldown=0,
                              patience=10, min_lr=10e-10,
                              verbose=1)

# TRAIN MODEL
epochs = 50
batch_size = 128
X = X_train
y = labelled_y_train

model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[history, checkpoint, csv_logger, reduce_lr, stop],
          verbose=1
          )

plot_losses(history)

plot_accuracy(history)

# RUN INFECTION CLASSIFIER IN PREDICTION MODE

## USE TRAINED MODEL TO RUN PREDICTIONS ON UNSEEN DATA
yhat = model.predict(X_test, verbose=1, batch_size=1)

print(pd.DataFrame(yhat, columns=['Normal', 'Inflammation', 'Infection', 'Severe inf', 'Unsure', 'Missing']).head(5))

# SHOW PREDICTION RESULTS
# Plot linewidth.
lw = 2
n_classes = 6

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_test = labelled_y_test
y_score = yhat

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC curve')
plt.legend(loc="lower right")
plt.show()
