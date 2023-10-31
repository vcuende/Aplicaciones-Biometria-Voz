import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import sys
import pathlib
import random
import librosa


ALS_table = pd.read_excel('ALS_table.xlsx', converters={'Subject code':str})
ALS_codes = ALS_table.get('Subject code')
HC_table = pd.read_excel('HC_table.xlsx', converters={'Subject code':str})
HC_codes = HC_table.get('Subject code')
data_dirALS = str(pathlib.Path('ALS'))
data_dirHC = str(pathlib.Path('HC'))
filenamesALS = tf.io.gfile.glob(data_dirALS + '/*')
filenamesHC = tf.io.gfile.glob(data_dirHC + '/*')
dataset_X = []
spectrograms = []
dataset_Y = []
dataset = []
HC_dataset = []
filenames = []
filenames = np.append(filenames, filenamesALS)
filenames = np.append(filenames, filenamesHC)
classes = ['ALS', 'HC']
filenames.sort()
print(len(filenamesALS))
print(len(filenamesHC))

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_mel_spectrogram(waveform):
  stfts = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrograms = tf.abs(stfts)
  num_spectrogram_bins = stfts.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20, 20000, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins, 44100.0, lower_edge_hertz, upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]
  return mfccs

def get_spectrogram_and_label(filepath):
  audioFile = tf.io.read_file(filepath)
  waveform = tf.audio.decode_wav(contents=audioFile, desired_samples=44100*3)
  waveform = tf.squeeze(waveform.audio, axis=-1)
  waveform = tf.cast(waveform, dtype=tf.float32)
  spectrogram = get_spectrogram(waveform)
  parts = tf.strings.split(
    input=filepath,
    sep=os.path.sep)
  label = tf.argmax(parts[-2] == classes)
  return spectrogram, label

def get_spectrogram_and_label(filepath):
  audioFile = tf.io.read_file(filepath)
  waveform = tf.audio.decode_wav(contents=audioFile, desired_samples=44100*3)
  waveform = tf.squeeze(waveform.audio, axis=-1)
  waveform = tf.cast(waveform, dtype=tf.float32)
  spectrogram = get_spectrogram(waveform)
  parts = tf.strings.split(
    input=filepath,
    sep=os.path.sep)
  label = tf.argmax(parts[-2] == classes)
  return spectrogram, label

def get_mel_spectrogram_and_label(filepath):
  audioFile = tf.io.read_file(filepath)
  waveform = tf.audio.decode_wav(contents=audioFile, desired_samples=44100*3)
  waveform = tf.squeeze(waveform.audio, axis=-1)
  waveform = tf.cast(waveform, dtype=tf.float32)
  spectrogram = get_mel_spectrogram(waveform)
  parts = tf.strings.split(
    input=filepath,
    sep=os.path.sep)
  label = tf.argmax(parts[-2] == classes)
  return spectrogram, label


iteraciones = 40
EPOCHS = 50
sizeTrain = 86
sizeVal = 18
sizeTest = 122 - sizeTrain - sizeVal
average = []
aciertoALS = []
aciertoHC = []
for iteracion in range(iteraciones):
    contTrain = 0
    contVal = 0
    contTest = 0
    trainFiles = [0] * sizeTrain
    valFiles = [0] * sizeVal
    testFiles = [0] * sizeTest
    noAdd = False
    contTestALS = 0
    contValALS = 0
    contFile = 0
    addTrain = True
    addVal = True
    addTest = True
    i = 0
    while(i < len(filenames)):
        n = random.randint(0,2)
        if(n==0):
            if(contTrain<sizeTrain and addTrain):
                trainFiles[contTrain] = filenames[i] 
                trainFiles[contTrain+1] = filenames[i+1] 
                contTrain += 2
                if(contTrain > (sizeTrain/2)+2 and contFile < 66):
                    addTrain = False
            else:
                noAdd = True
        elif(n==1):
            if(contVal<sizeVal and addVal):
                valFiles[contVal] = filenames[i] 
                valFiles[contVal+1] = filenames[i+1] 
                contVal += 2
                if(contVal > sizeVal/2 and contFile < 66):
                    addVal = False
            else:
                noAdd = True
        elif(n==2):
            if(contTest<sizeTest and addTest):
                testFiles[contTest] = filenames[i] 
                testFiles[contTest+1] = filenames[i+1] 
                contTest += 2
                if(contTest > sizeTest/2 and contFile < 66):
                    addTest = False
            else:
                noAdd = True
        if(noAdd):
            if(contTrain<sizeTrain and addTrain):
                trainFiles[contTrain] = filenames[i] 
                trainFiles[contTrain+1] = filenames[i+1] 
                contTrain += 2
            elif(contVal<sizeVal and addTest):
                valFiles[contVal] = filenames[i] 
                valFiles[contVal+1] = filenames[i+1] 
                contVal += 2
            elif(contTest<sizeTest and addVal):
                testFiles[contTest] = filenames[i] 
                testFiles[contTest+1] = filenames[i+1] 
                contTest += 2
        noAdd = False
        i += 2
        contFile += 2
        if (contFile>66 and ((not addTrain) or (not addVal) or (not addTest))):
            addTrain = True
            addVal = True
            addTest = True
    trainFiles = tf.random.shuffle(trainFiles)
    testFiles = tf.random.shuffle(testFiles)
    valFiles = tf.random.shuffle(valFiles)
    AUTOTUNE = tf.data.AUTOTUNE
    # Train subdataset
    files_ds = tf.data.Dataset.from_tensor_slices(trainFiles)
    spectrogram_ds = files_ds.map(map_func=get_mel_spectrogram_and_label,num_parallel_calls=AUTOTUNE)
    # Test subdataset
    files_ds = tf.data.Dataset.from_tensor_slices(testFiles)
    test_ds = files_ds.map(map_func=get_mel_spectrogram_and_label,num_parallel_calls=AUTOTUNE)
    # Validation subdataset
    files_ds = tf.data.Dataset.from_tensor_slices(valFiles)
    val_ds = files_ds.map(map_func=get_mel_spectrogram_and_label,num_parallel_calls=AUTOTUNE)
    train_ds = spectrogram_ds
    train_ds = train_ds.batch(4)
    val_ds = val_ds.batch(4)
    #AÃ±adiendo cache para reducir latencia
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    for spectrogram, label in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        # Normalize.
        norm_layer,
        layers.Conv1D(8, 3, activation='relu', kernel_regularizer='l2'),
        layers.Conv1D(16, 3, activation='relu', kernel_regularizer='l2'),
        layers.Conv1D(32, 3, activation='relu', kernel_regularizer='l2'),
        layers.MaxPooling1D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(32, activation='relu', kernel_regularizer='l1'),
        layers.Dense(32, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.6),
        layers.Dense(2),
    ])
    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, patience=15),
    )
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    average.append(test_acc)
    print(f'Test set accuracy: {test_acc:.0%}')
    metrics = history.history
    # plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    # plt.legend(['loss', 'val_loss'])
    # plt.show()
    # plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    # plt.legend(['accuracy', 'val_accuracy'])
    # plt.show()
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    matriz = confusion_mtx.numpy()
    aciertoALS.append(matriz[0][0])
    aciertoHC.append(matriz[1][1])
    # plt.figure(figsize=(10,8))
    # sns.heatmap(confusion_mtx,
    # xticklabels=classes,
    # yticklabels=classes,
    # annot=True, fmt='g')
    # plt.xlabel('Prediction')
    # plt.ylabel('Label')
    # plt.show()
    print("Iteracion:", iteracion)
print('Average accuracy:' , np.mean(np.array(average)))
print('Average accuracy ALS:', np.mean(aciertoALS)*10)
print('Average accuracy HC:', ((np.sum(aciertoHC)/iteraciones)/8)*100)