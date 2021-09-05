from csv import reader
from classes import poi,pois, interface, scaleData
from math import ceil,floor
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

def getSample(data,rate=30,k=10080,n=1440):
    #k denotes window for training samples, n denotes window for evaluation samples
    # k >> n
    if (k < n):
        raise Exception("k must be larger than n")
    # 0:timestamp 1:open 2:high 3:low 4:close
    xo = randint(k,len(data)-n)

    open,close = zeros(len(data)), zeros(len(data))
    for i in range(len(data)):
        open[i],close[i] = data[i][1],data[i][4]

    trainSample = pois(open[xo-k:xo],close[xo-k:xo],rate)
    evalSample = pois(open[xo:xo+n],close[xo:xo+n],rate)

    return trainSample,evalSample


def getSamples(datafile,nSamples = 100,rate = 30,k=10080,n=1440):
    #k denotes window for training samples, n denotes window for evaluation samples
    # k >> n
    if (k < n):
        raise Exception("k must be larger than n")
    data = getData(datafile)

    trainingSamples,evaluationSamples = list(), list()
    for i in range(nSamples):
        train,eval = getSample(data,rate,k,n)
        trainingSamples.append(train)
        evaluationSamples.append(eval)

    if nSamples == 1:
        trainingSamples[0].addPoints()
        evaluationSamples[0].addPoints()
    else:
        pool = Pool(min([6,nSamples]))
        trainingSamples = pool.map(pois.addPoints,trainingSamples)
        evaluationSamples = pool.map(pois.addPoints,evaluationSamples)
    
        pool.close()
        pool.join()

    return trainingSamples,evaluationSamples

def subsample(sample,nSamples=1e4,window_size=360,pred_len=5):
    xData,yData = list(),list()
    #xRef,yRef = list(),list()

    def getSubsample(sample):
        xo = randint(0,len(sample.close)-1-window_size-pred_len)
        #lstm input dimensions are [ batch_size, timesteps, channels ]
        xData.append(reshape(scaleData(sample.zeromean[xo:xo+window_size]),(window_size,1)))
        #xRef.append(sample.data[xo:xo+window_size])
        #yRef.append(sample.binrep[xo:xo+window_size])
        xo += pred_len
        yData.append(reshape(sample.binrep[xo:xo+window_size],(window_size,1)))

    for i in range(int(nSamples)):
        getSubsample(sample)

    return xData,yData#,xRef,yRef

def cycle(data,npt):
    data = data[1:]
    return append(data,npt)

def evaluate(sample,model,window_size,prediction_length):
    trader = interface(windowSize=window_size, predLen=prediction_length)
    trader.model = model

    delta,entry,exit,wins,losses,temp = 0,0,0,0,0,0
    total = 1
    ls = None
    wOpen,wClose = zeros((window_size)), zeros((window_size))

    for i in range(len(sample.close)):
        if i % 1440 == 0:
            print("\nday ",int(i/1440) + 1,"\n")

        wOpen, wClose = cycle(wOpen,sample.open[i]), cycle(wClose,sample.close[i])
        if wOpen[0] == 0:
            continue

        #clean this up
        ls,exit = trader.iter(wOpen,wClose)
        
        if ls is not None:
            if entry == 0:
                entry = exit
            elif ls:
                temp = entry-exit
                temp = temp/entry

                if temp > 0:
                    wins += 1
                else: 
                    losses += 1

                total = total*(1+temp)
                delta += temp
                entry = exit
                print("minute: ",i+1," long , total delta: %", delta, "   accuracy : %", int(100*wins/(wins+losses)))
                print("exp_total: ",total)
            elif not ls:
                temp = exit-entry
                temp = temp/entry
                
                if temp > 0:
                    wins += 1
                else: 
                    losses += 1
                
                total = total*(1+temp)
                delta += temp
                entry = exit
                print("minute: ",i+1," short, total delta: %", delta, "   accuracy : %", int(100*wins/(wins+losses)))
                print("exp_total: ",total)
    
    print("\n\nTotal delta: %", delta, "   accuracy : %", int(100*wins/(wins+losses)))
    print("Total Trades: ",wins+losses)
    print("exp_total: ",total)

    return delta, wins, losses, total

