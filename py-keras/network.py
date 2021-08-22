from functions import getSamples,subsample
from numpy import transpose
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense, Activation, Dropout


def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    return Model([
        Input(shape=(1,windowSize), batch_size = batchSize),

        Bidirectional(LSTM(hiddenLayers)),
        Dropout(0.2),
        Activation('relu'),
        Dense(windowSize),

        Bidirectional(LSTM(hiddenLayers)),
        Dropout(0.2),
        Activation('relu'),

        Bidirectional(LSTM(hiddenLayers)),
        Dropout(0.2),
        Activation('relu'),

        Dense(1)
        ])


if __name__ == '__main__':  

    hL = 128 #hidden layers
    Ws = 360 #windowSize
    batchSize = 64
    nSamples = 1e4
    predLen = 5


    samples = getSamples("BTCUSDT.txt")
    xData,yData = subsample(samples,nSamples,Ws,predLen)
    print(xData[1].shape,transpose(yData[1].shape))
    
    '''
    model = getModel(hL,batchSize,Ws)
    model.compile(optimizer='adam',loss="mean_squared_error")
    model.fit(xData,yData,
              batch_size=batchSize,
              epochs=100,
              validation_split=0.2
              )

    '''
 
    