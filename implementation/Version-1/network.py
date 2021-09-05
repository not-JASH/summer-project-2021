from functions import getSamples,subsample, evaluate
import time
from numpy import array,zeros
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout, Conv1DTranspose, Conv1D, Conv2DTranspose, Conv2D, Reshape
from keras.engine.input_layer import InputLayer as Input 
from keras.callbacks import EarlyStopping

def getModel(hiddenLayers = 128, batchSize = 32, windowSize = 100):
    return Sequential([
        Input(input_shape=(windowSize,1)),

        LSTM(int(hiddenLayers),return_sequences=True),
        Dropout(0.2),
        Activation('relu'),

        LSTM(int(hiddenLayers),return_sequences=True),
        Dropout(0.2),
        Activation('relu'),

        LSTM(int(hiddenLayers),return_sequences=True),
        Dropout(0.2),
        Activation('relu'),

        LSTM(int(hiddenLayers),return_sequences=True),
        Dropout(0.2),
        Activation('relu'),
       
        Dense(1)
        ])

def trainModel(model,batch_size=None,xData=None,yData=None,learn_rate=None,no_samples=None,validSamples=None):
       model.compile(
            optimizer=Adam(
                ExponentialDecay(learn_rate,
                                 int(no_samples*20),
                                 0.5
                                 )
                ),
            loss=MeanSquaredError()
            )
       model.fit(xData,yData,batch_size,
                  epochs=100,
                  #validation_split=0.2,
                  validation_data = validSamples,
                  shuffle=True,
                  verbose=2,
                  callbacks=[
                      EarlyStopping(
                      #monitor='val_loss',
                      monitor='loss',
                      patience=4,
                      mode='min',
                      verbose=0
                      )]
                  )
       return model

if __name__ == '__main__':  

    model_name = "v2-"
    model_name = "/implementation/version-1/" + model_name

    hidden_layers = 64
    window_size = 60
    batch_size = 250
    no_samples = 1e4
    no_sets = 10
    prediction_length = 1
    k = 24*60
    n = int(4*60)

    learn_rate = 2e-3

    trainSamples,evalSamples = getSamples("BTCUSDT.txt",nSamples=no_sets,rate=5,k=k,n=n)

    results = zeros((no_sets,4))
    for i in range(no_sets):
        xData,yData = subsample(trainSamples[i],nSamples=no_samples,window_size=window_size,pred_len=prediction_length)
        xData,yData = array(xData), array(yData)
        
        xVal, yVal = subsample(evalSamples[i],nSamples=no_samples,window_size=window_size,pred_len=prediction_length)
        xVal,yVal = array(xVal), array(yVal)

        start = time.time()
        model = getModel(hidden_layers,batch_size,window_size)
        print(model.summary())

        model = trainModel(model,
                           batch_size=batch_size,
                           xData=xData,
                           yData=yData,
                           learn_rate=learn_rate,
                           no_samples=no_samples,
                           validSamples=(xVal,yVal)
                           )

        print("time elapsed: ", time.time()-start)

        results[i][0], results[i][1], results[i][2], results[i][3] = evaluate(evalSamples[i],model,window_size,prediction_length)
        
        print(results)