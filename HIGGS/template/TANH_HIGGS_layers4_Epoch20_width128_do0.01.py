# UCI Machine Learning Repository


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

import pandas as pd
import time
import sys

# Hyper-parameters
# 	nb of hidden layers
layers = 4
EPOCH = 20
width = 128
dropout = .01
batch_size = 100
learning_rate = .01
# ------------------------------------------------- #


benchmark = str(sys.argv[1])  # highlevel or all
# input_n is the number of features input according to each type of benchmark(raw, highlevel, all)
input_n = 0

# Column to load according to benchmark
# data range is 0 to 28 for Higgs.
# 21 low-level features & 7 high-level features
# and the column 0 is for the label
all_n = 28  # total features
raw_n = 21  # raw features
high_n = 7  # highlevel features

# validation split: number of instances used for validation

validation_split = 20 / 100


# list(range(a, b+1)) returns an array of [a, ..., b]
# usecols points to the columns to use
if benchmark == "raw":
    # [1, .., 21] ===> 21 items
    # ===> [0,1, ..., 21]
    usecols = [0] + list(range(1, raw_n + 1))
    input_n = raw_n
elif benchmark == "highlevel":
    # [22, ..., 28] ===> 7 items
    # [0,22, ..., 28]
    usecols = [0] + list(range(raw_n + 1, all_n + 1))
    input_n = high_n
elif benchmark == "all":
    # [1, ..., 28] ===> 28 items
    # [0,1, ..., 28]
    usecols = [0] + list(range(1, all_n + 1))
    input_n = all_n

print(benchmark)


# --------------------------------


# Construct the model graph
def model():

    m = Sequential()
    activation_fn = tf.nn.relu

    # hidden layers & input layer
    m.add(Dense(width, activation=activation_fn, input_shape=(input_n,)))
    m.add(Dropout(dropout))
    m.add(Dense(width, activation=activation_fn))
    m.add(Dropout(dropout))
    m.add(Dense(width, activation=activation_fn))
    m.add(Dropout(dropout))
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
datas = pd.read_csv(filepath_or_buffer=r"F:\Documents\Memoir\Datas\HIGGS.csv.gz",
                    low_memory=True, compression="gzip", usecols=usecols,
                    na_filter=False)
print("Loading Time : {:6.6}s".format(time.time() - t))

# Data preprocessing

# .iloc[:, a:b] means all datas in the 1st dimension and the datas which index is between 'a and b-1' of in the 2nd dimension
x = datas.iloc[:, 1:].as_matrix()
y = datas.iloc[:, 0].as_matrix()  # returns a 1D-array


# #   Encode the data to int if needed
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_y = encoder.transform(y)

# Build the model
# 	Filename
home = 'F:\\Documents\\Memoir\\Datas\\process'
log_dir = home + '\\logs\\HIGGS\\TANH_model_HIGGS_layers{}_Epoch{}_width{}_do{}_{}'
log_dir = log_dir.format(layers, EPOCH, width, dropout, benchmark)
filename = home + '\\saves\\HIGGS\\TANH_model_HIGGS_layers{}_Epoch{}_width{}_do{}_{}.h5'
filename = filename.format(layers, EPOCH, width, dropout, benchmark)

# 	Callbacks
tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=500)
save = keras.callbacks.ModelCheckpoint(filename, verbose=1, save_best_only=True, save_weights_only=False, period=1)

# 	model.fit(x_train, y_train, .....)
model.fit(x, y, batch_size=batch_size, verbose=1,
          validation_split=validation_split, nb_epoch=EPOCH, callbacks=[tb, early_stop, save])
