from functions import getSamples,subsample, evaluate
from numpy import array
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout
from keras.engine.input_layer import InputLayer as Input 
from keras.callbacks import EarlyStopping

def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    #batch size has been omitted from this version's input layer
    return Sequential([
        Input(input_shape=(windowSize,1)),

        Bidirectional(LSTM(hiddenLayers,return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),
        Dense(1),

        Bidirectional(LSTM(int(0.5*hiddenLayers),return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),

        Bidirectional(LSTM(hiddenLayers,return_sequences=True)),
        Dropout(0.2),
        Activation('relu'),

        Dense(1)
        ])

if __name__ == '__main__':  

    model_name = "v2-"
    model_name = "/implementation/version-1/" + model_name

    hidden_layers = 128
    window_size = 360
    batch_size = 100
    no_samples = 1e4
    no_sets = 10
    prediction_length = 5
    k = 10080
    n = 1440

    learn_rate = 1e-3

    trainSamples,evalSamples = getSamples("BTCUSDT.txt",nSamples=no_sets,rate=30,k=k,n=n)

    for i in range(no_sets):
        xData,yData = subsample(trainSamples[i],nSamples=no_samples,window_size=window_size,pred_len=prediction_length)
        xData,yData = array(xData), array(yData)
        #print(xData.shape,yData.shape)

        model = getModel(hidden_layers,batch_size,window_size)
        model.compile(
            optimizer=Adam(learn_rate),
            loss=MeanSquaredError()
            )
        model.fit(xData,yData,batch_size,
                  epochs=100,
                  validation_split=0.2,
                  shuffle=True,
                  verbose=2,
                  callbacks=[
                      EarlyStopping(
                      monitor='val_loss',
                      patience=4,
                      mode='min',
                      verbose=0
                      )]
                  )
        model.save((model_name + str(i)))

        evaluate(evalSamples[i],model,window_size)
        
