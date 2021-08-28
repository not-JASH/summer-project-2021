from sys import exit,argv
from getopt import getopt, GetoptError
from functions import getSamples,subsample
from numpy import array
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout
from keras.engine.input_layer import InputLayer as Input 


def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    return Sequential([
        Input(input_shape=(1,windowSize)),

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

        Dense(windowSize)
        ])

def getArgs():
    hiddenLayers = 128
    windowSize = 360
    batchSize = 100
    nSamples = 6.4*1e4
    predLen = 5

    help = "network.py -l <hidden layers> -w <window size> -b <batch size> -s <no. samples> -p <prediction length>"

    try:
        opts,args = getopt(argv,"hl:w:b:s:p:")
        #skipped long_options :)

    except GetoptError:
        print(help)
        exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(help)
            exit()
        elif opt == "-l":
            hiddenLayers = arg
        elif opt == "-w":
            windowSize = arg
        elif opt == "-b":
            batchSize = arg
        elif opt == "-s":
            nSamples = arg
        elif opt == "p":
            predLen = arg

    return hiddenLayers,windowSize,batchSize,nSamples,predLen




if __name__ == '__main__':  

    hL,Ws,batchSize,nSamples,predLen = getArgs()

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
        epochs=1,
        validation_split=0.2,
        shuffle=True
        )
    
    model.save("version1")
 
    
