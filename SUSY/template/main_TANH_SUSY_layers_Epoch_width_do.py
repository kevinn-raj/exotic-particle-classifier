# UCI Machine Learning Repository


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

import pandas as pd
import time
import sys


def train_SUSY(layers=4, EPOCH=200, width=128, dropout=0.5, benchmark="raw",
               top=False):
    """
    # Hyper-parameters
    layers : number of hidden layers
    EPOCH : number of training epochs
    width : number of units for each hidden layer
    dropout : dropout probability
    benchmark : the type of datasets to be used raw, highlevel or all
    """
    # input_n is the number of features input according to each type of 
    # benchmark(raw, highlevel, all)
    input_n = 0



    # Number of column to load according to benchmark
    # data range is 0 to 18 for SUSY.
    # 8 low-level features & 10 high-level features
    # and the column 0 is for the label
    all_n = 18  # total features
    raw_n = 8  # raw features
    high_n = 10  # highlevel features

    # validation split: number of instances used for validation
    validation_split = 20 / 100


    # list(range(a, b+1)) returns an array of [a, ..., b]
    # usecols points to the columns to use
    usecols = []
    if benchmark == "raw":
        # [1, .., 8] ===> 8 items
        usecols = [0] + list(range(1, raw_n + 1))
        input_n = raw_n
    elif benchmark == "highlevel":
        # [9, ..., 18] ===> 10 items
        usecols = [0] + list(range(raw_n + 1, all_n + 1))
        input_n = high_n
    elif benchmark == "all":
        # [1, ..., 18] ===> 18 items
        usecols = [0] + list(range(1, all_n + 1))
        input_n = all_n

    print('benchmark: ', benchmark)
    print("usecols: ", usecols)

    # --------------------------------


    # Construct the model graph
    def model():

        m = Sequential()
        activation_fn = tf.nn.relu

        # hidden layers & input layer
        m.add(Dense(width, activation=activation_fn, input_shape=(input_n,)))
        m.add(Dropout(dropout))

        for i in range(0, layers-1):
            m.add(Dense(width, activation=activation_fn))
            m.add(Dropout(dropout))

        # output layer
        m.add(Dense(1, activation=activation_fn))
        m.summary()
        return m

    model = model()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # -------------------------------------------------------- #

    # Data loading
    print("Loading data")
    t = time.time()
    tables = []

    # Path of the compressed dataset
    path = r"G:\Documents\Memoir\Datas\SUSY.csv.gz"

    datas = pd.read_csv(filepath_or_buffer=path,
                        low_memory=True, compression="gzip", usecols=usecols,
                        na_filter=False)
    print("Loading Time : {:6.6}s".format(time.time() - t))

    # Data preprocessing

    # Slicing the data
    x = datas.iloc[:, 1:].as_matrix()
    y = datas.iloc[:, 0].as_matrix()  # returns a 1D-array


    # #   Encode the data to int if needed
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # encoded_y = encoder.transform(y)

    # Build the model
    # 	Filename
    home = '../../../Process'
    log_dir = home + '\\logs\\SUSY\\TANH_model_SUSY_layers{}_Epoch{}_width{}_do{}_{}'
    log_dir = log_dir.format(layers, EPOCH, width, dropout, benchmark)
    filename = home + '\\saves\\SUSY\\TANH_model_SUSY_layers{}_Epoch{}_width{}_do{}_{}.h5'
    filename = filename.format(layers, EPOCH, width, dropout, benchmark)

    # 	Callbacks
    tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, 
                                     write_graph=True, write_images=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=500)
    save = keras.callbacks.ModelCheckpoint(filename, verbose=1, 
                        save_best_only=True, save_weights_only=False, period=1)

    # 	model.fit(x_train, y_train, .....)
    model.fit(x, y, batch_size=batch_size, verbose=1,
              validation_split=validation_split, nb_epoch=EPOCH, 
              callbacks=[tb, early_stop, save])
