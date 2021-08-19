from classes import poi, pois
from keras.models import model
from keras.layers import Input, LSTM, Bidirectional, Dense, Activation, Dropout


hL = 128 #hidden layers
Ws = 360 #windowSize
batchSize = 64

model = model([
    Input(shape=(1,Ws), batch_size = batchSize),

    Bidirectional(LSTM(hL)),
    Dropout(0.2),
    Activation('relu'),
    Dense(Ws),

    Bidirectional(LSTM(hL)),
    Dropout(0.2),
    Activation('relu'),

    Bidirectional(LSTM(hL)),
    Dropout(0.2),
    Activation('relu'),

    Dense(1)
    ])


 
