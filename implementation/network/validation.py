#from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots,plot,show
from functions import getSamples,subsample
from keras.models import load_model
from numpy import reshape
from os import getcwd
import numpy


def dualplot(data1,data2):
    figure,axis1 = subplots()
    axis1.plot(data1)

    axis2 = axis1.twinx()
    axis2.plot(data2)

    show()



if __name__ == '__main__':
    path = 'version1-1'
    #path = getcwd() + '/implementation/trade-interface/' + path
    model = load_model(path)

    samples = getSamples("BTCUSDT.txt")
    xData,yData,xRef,yRef = subsample(samples,1e3,360,5)
 
    prediction = model.predict(reshape(xData[1],(1,1,360)))
    prediction = reshape(prediction,(360))
    
    dualplot(numpy.round(prediction),reshape(yData[1],(360)))
   


