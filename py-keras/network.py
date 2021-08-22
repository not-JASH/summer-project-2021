from functions import getSamples,subsample
from numpy import array
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout
from keras.engine.input_layer import InputLayer as Input 


def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    return Sequential([
        Input(input_shape=(windowSize,1,), batch_size = batchSize),

        Bidirectional(LSTM(hiddenLayers,return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),
        Dense(windowSize),

        Bidirectional(LSTM(hiddenLayers,return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),

        Bidirectional(LSTM(hiddenLayers,return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),

        Dense(1)
        ])


if __name__ == '__main__':  

    hL = 128 #hidden layers
    Ws = 360 #windowSize
    batchSize = 64
    nSamples = 3.2*1e4
    predLen = 5


    samples = getSamples("BTCUSDT.txt")
    xData,yData = subsample(samples,nSamples,Ws,predLen)

    xData = array(xData)
    yData = array(yData)
    print(xData.shape,yData.shape)

    model = getModel(hL,batchSize,Ws)
    
    model.compile(
        optimizer=Adam(learning_rate = 1e-3),
        loss=MeanSquaredError()
        )  
    
    model.fit(
        xData,
        yData,
        batch_size=batchSize,
        epochs=100,
        validation_split=0.2,
        shuffle=True
        )
    
    model.save("version1")

    