#from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots,plot,show
from functions import getSamples,subsample


def dualplot(data1,data2):
    figure,axis1 = subplots()
    axis1.plot(data1)

    axis2 = axis1.twinx()
    axis2.plot(data2)

    show()



if __name__ == '__main__':

    samples = getSamples("BTCUSDT.txt")
    xData,yData,xRef,yRef = subsample(samples,1e3,360,5)
 
    dualplot(xRef[1],yRef[1])

   


