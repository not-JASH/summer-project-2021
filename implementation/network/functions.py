from csv import reader
from classes import poi,pois
from math import ceil
from random import randint
from multiprocessing import Pool
from numpy import zeros,append,linspace, round,reshape

def getData(datafile):
    '''
    read from file or download from binance?
    '''

    with open(datafile) as file:
        data = list(reader(file, delimiter=' '))

    return data

def getSamples(datafile,nSamples = 100,rate = 30):
    data = getData(datafile)
    # 0:timestamp 1:open 2:high 3:low 4:close
    close = zeros(len(data))
    open = zeros(len(data))
    for i in range(len(data)):
        close[i] = data[i][4]
        open[i] = data[i][1]

    data = close
    samples = list();
    #limits = []
    limits = round(linspace(1,len(close),nSamples+1))

    for i in range(nSamples):
        samples.append(pois(open[int(limits[i]):int(limits[i+1])],close[int(limits[i]):int(limits[i+1])],rate))

    pool = Pool(6)
    samples = pool.map(pois.addPoints,samples)    
    

    pool.close()
    pool.join()

    return samples

def subsample(samples,nSamples = 1e4,windowSize = 360, predLen = 5):
    sampleLocs = [randint(0,len(samples)-1) for i in range(ceil(nSamples))]
    xData = list()
    yData = list()
    xRef = list()
    yRef = list()

    def getSubsample(sample):
        xo = randint(0,len(sample.data)-1-windowSize-predLen)
        xData.append(reshape(scaleData(sample.zeromean[xo:xo+windowSize]),(1,windowSize)))
        xRef.append(sample.data[xo:xo+windowSize])
        yRef.append(sample.binrep[xo:xo+windowSize])
        xo += predLen
        yData.append(reshape(sample.binrep[xo:xo+windowSize],(1,windowSize)))
        
    for i in range(len(sampleLocs)):
        getSubsample(samples[sampleLocs[i]])

    return xData,yData,xRef,yRef


def scaleData(data):
    ''' 
    would it be better to normalize data by mean and variance

    '''

    data = data-min(data)
    data = data/max(data)
    data = 2*data - 1
  
    return data






    