from os import getcwd
from csv import reader
from numpy import zeros, append
from classes import interface

def getData(datafile):
    '''
    read from file or download from binance?
    '''

    with open(datafile) as file:
        data = list(reader(file, delimiter=' '))

    return data

def prepareData(data):
    close = zeros(len(data))
    open = zeros(len(data))

    for i in range(len(data)):
        close[i] = float(data[i][4])
        open[i] = float(data[i][1])

    return open, close

def modelPath():
    modelpath = 'v2-3'
    #modelpath = getcwd() + '/implementation/trade-interface/' + modelpath
    
    return modelpath

def dataPath():
    datapath = "BTCUSDT_1.txt"
    datapath = getcwd() + '/../network/' + datapath

    return datapath

def cycle(data,npt):
    data = data[1:]
    return append(data,npt)

if __name__ == '__main__':  
    ws = 360
    trader = interface(windowSize=ws)
    trader.loadModel(modelPath())
    
    open,close = prepareData(getData(dataPath()))
    delta,entry,exit,wins,losses,temp = 0,0,0,0,0,0
    ls = None
    wOpen = zeros((ws))
    wClose = zeros((ws))

    for i in range(len(close) - ws):
    #for i in range(60*24+0*60*24):
        if i % 1440 == 0:
            print("\n\n\n",int(i/1440)," days\n\n")
        
        wOpen = cycle(wOpen,open[i])
        wClose = cycle(wClose,close[i])

        if wOpen[0] == 0:
            continue
        
        ls,exit = trader.iter(wOpen,wClose)
        if ls is not None:
            if entry == 0:
                entry = exit
            elif ls:
                temp = entry-exit
                
                if temp > 0:
                    wins += 1
                else: 
                    losses += 1

                delta += temp
                entry = exit
                print("long, total delta: ", delta, "   accuracy : ", wins/(wins+losses))
            elif not ls:
                temp = exit-entry
                
                if temp > 0:
                    wins += 1
                else: 
                    losses += 1
                
                delta += temp
                entry = exit
                print("short, total delta: ", delta, "   accuracy : ", wins/(wins+losses))

    print("\n\nTotal delta: ", delta, "   accuracy : ", wins/(wins+losses))
