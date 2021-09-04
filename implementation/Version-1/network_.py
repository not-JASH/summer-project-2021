from functions import getSamples,subsample, evaluate, getBatches, combine
from transformer import Transformer
import time
from numpy import array,zeros
from keras.models import Model, Sequential 
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.layers import LSTM, Bidirectional, Dense, Activation, Dropout, Conv1DTranspose, Conv1D, Conv2DTranspose, Conv2D, Reshape
from keras.engine.input_layer import InputLayer as Input 
from keras.callbacks import EarlyStopping

if __name__ == '__main__':  

    model_name = "transformer-"
    model_name = "/implementation/version-1/" + model_name

    hidden_layers = 64
    window_size = 60
    batch_size = 50
    no_samples = 1e4
    no_sets = 10
    prediction_length = 1
    k = 24*60
    n = int(4*60)

    learn_rate = 2e-3

    trainSamples,evalSamples = getSamples("BTCUSDT.txt",nSamples=no_sets,rate=5,k=k,n=n)

    results, history = zeros((no_sets,4)), []

    for i in range(no_sets):
        xData,yData = subsample(trainSamples[i],nSamples=no_samples,window_size=window_size,pred_len=prediction_length)
        xData,yData = array(xData), array(yData)
        
        xVal, yVal = subsample(evalSamples[i],nSamples=int(0.2*no_samples),window_size=window_size,pred_len=prediction_length)
        xVal,yVal = array(xVal), array(yVal)

        xData, yData = getBatches(xData,batch_size), getBatches(yData,batch_size)
        xVal, yVal = getBatches(xVal,batch_size), getBatches(yVal,batch_size)

        start = time.time()
        model = Transformer(input_size=window_size,
                            target_size=window_size,
                            no_layers=4,
                            d_model=64,
                            dff=64,
                            no_heads=8,
                            dropout_rate=0.1                            
                            )

        history.append(model.train(combine(xData,yData),combine(xVal,yVal),batch_size))
        
        print("time elapsed: ", time.time()-start)

        results[i][0], results[i][1], results[i][2], results[i][3] = evaluate(evalSamples[i],model,window_size)
        
        print(results)