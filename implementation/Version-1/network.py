from functions import getSamples,subsample, evaluate
from numpy import array
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout, Conv1DTranspose, Conv1D, Conv2DTranspose, Conv2D, Reshape
from keras.engine.input_layer import InputLayer as Input 
from keras.callbacks import EarlyStopping

def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    #batch size has been omitted from this version's input layer
    return Sequential([
        Input(input_shape=(windowSize,1)),

        Bidirectional(LSTM(int(hiddenLayers),
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        dropout=0,
                        return_sequences=True
                        )),
        Activation('relu'),
        Dropout(0.2),

        Bidirectional(LSTM(int(hiddenLayers),
                           dropout=0.0,
                           activation='tanh',
                           recurrent_activation='sigmoid',
                           return_sequences=True
                           )),
        Activation('relu'),

        Dense(1)
        ])

if __name__ == '__main__':  

    model_name = "v2-"
    model_name = "/implementation/version-1/" + model_name

    hidden_layers = 64
    window_size = 90
    batch_size = 100
    no_samples = 1e4
    no_sets = 10
    prediction_length = 0
    k = 14*24*60
    n = int(24*60)

    learn_rate = 1e-3

    trainSamples,evalSamples = getSamples("BTCUSDT.txt",nSamples=no_sets,rate=30,k=k,n=n)

    for i in range(no_sets):
        xData,yData = subsample(trainSamples[i],nSamples=no_samples,window_size=window_size,pred_len=prediction_length)
        xData,yData = array(xData), array(yData)
        #print(xData.shape,yData.shape)

        model = getModel(hidden_layers,batch_size,window_size)
        model.compile(
            optimizer=Adam(
                ExponentialDecay(learn_rate,
                                 int(no_samples*3),
                                 0.8
                                 )
                ),
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
                      #monitor='loss',
                      patience=4,
                      mode='min',
                      verbose=0
                      )]
                  )
        #model.save((model_name + str(i)))

        evaluate(evalSamples[i],model,window_size)
        
