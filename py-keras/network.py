from functions import getSamples,subsample
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense, Activation, Dropout

if __name__ == '__main__':  

    hL = 128 #hidden layers
    Ws = 360 #windowSize
    batchSize = 64
    nSamples = 1e4
    predLen = 5


    samples = getSamples("BTCUSDT.txt")
    xData,yData = subsample(samples,nSamples,Ws,predLen)

    print(xData[1].shape,yData[1].shape)

    '''

    model = Model([
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


 
    '''