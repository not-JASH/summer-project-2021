#from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as rmse
from matplotlib.pyplot import subplots,plot,show, axvline
from functions import getSamples,subsample
from keras.models import load_model
from numpy import reshape
from os import getcwd
import numpy


def dualplot(data1,data2):
    figure,axis1 = subplots()
    axis1.plot(data1)

    axis2 = axis1.twinx()
    axis2.plot(data2,'m')

    axvline(355,color='r')    
    show()



if __name__ == '__main__':
    path = 'v2-3'
    #path = getcwd() + '/implementation/trade-interface/' + path
    model = load_model(path)

    samples = getSamples("BTCUSDT.txt")
    xData,yData,xRef,yRef = subsample(samples,1e3,360,5)

    '''
    for i in range(len(xData)):
        prediction = model.predict(reshape(xData[i],(1,360,1)))

        prediction = reshape(prediction,(360))
        #prediction = numpy.round(prediction)
    
        dualplot(prediction,reshape(yData[i],(360)))

        #input("press enter to continue")
   '''
    error = 0
    prediction_error = 0
    for i in range(len(xData)):
        prediction = model.predict(reshape(xData[i],(1,360,1)))
        prediction = reshape(prediction,(360))
        #prediction = numpy.round(prediction)
        
        error += rmse(reshape(yData[i],(360)),prediction,squared=True)
        prediction_error += rmse(reshape(yData[i],(360))[-6:],prediction[-6:],squared=True)


    print("error",error/len(xData))
    print("prediction error",prediction_error/len(xData))



        


    


