import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as st
import pywt
import scipy.stats
import datetime as dt
from collections import defaultdict, Counter
from tensorflow import keras

def preprocessClassify(datafile):
    SAMPLE_RATE = 128
    MAX_DURATION = 30

        #start from here
    data = pd.read_csv(datafile, names = list(range(0,138)), low_memory=False)
    data = data.drop(data.index[0:2])
    data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)
    data = data.drop(data.columns[14:], axis=1)
    data = data.swapaxes("index", "columns")
    data = data.to_numpy()
    values = []
    for channel in data:
        for i in range(15):
            if channel[i*SAMPLE_RATE*MAX_DURATION:(i+1)*SAMPLE_RATE*MAX_DURATION].shape[0] == SAMPLE_RATE*MAX_DURATION:
    #                print(channel[i*SAMPLE_RATE*MAX_DURATION:(i+1)*SAMPLE_RATE*MAX_DURATION].shape)
                    values.append(channel[i*SAMPLE_RATE*MAX_DURATION:(i+1)*SAMPLE_RATE*MAX_DURATION])


    #feature extraction
    x = get_eeg_features(values, 'sym5') 
    x = pd.DataFrame(x)
    #X.shape

    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x)
    x = scaler.transform(x)
    #x.shape
    
    #ANN
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization,LeakyReLU
    
    from tensorflow.keras.models import Sequential,load_model 
    from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
    from tensorflow.keras import initializers


    ann = Sequential()
    ann.add(Dense(units = 108,  input_dim =108))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(BatchNormalization(momentum=0.8))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 256, ))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 128,))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 128,))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 128,))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 64))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 64))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 32))
    ann.add(LeakyReLU(alpha=0.2))
    ann.add(Dropout(0.5))

    ann.add(Dense(units = 2, activation = 'sigmoid'))

    ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'], run_eagerly=True)
                
    ann_output = np.argmax(ann.predict(x), axis=-1)
    prediction = st.mode(ann_output)

    return prediction[0][0]

    

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

    #feature extraction
def get_eeg_features(eeg_data, waveletname):
    list_features = []
#   list_unique_labels = list(set(ecg_labels))
#   list_labels = [list_unique_labels.index(elem) for elem in ecg_labels]
    for signal in eeg_data:
        list_coeff = pywt.wavedec(signal, waveletname)
        features = []
        for coeff in list_coeff:
            features += get_features(coeff.flatten()) 
        list_features.append(features)
    return list_features