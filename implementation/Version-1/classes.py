from tensorflow import keras
from keras.models import load_model
from binance.client import Client
from numpy import argmax, argmin, zeros, reshape

class poi:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getSecant(self,rhn):
        m = (rhn.y - self.y)/(rhn.x - self.x)
        b = self.y-(m*self.x);
        return [m,b]

    def getMidPoint(self,rhn,data):
        
        coeffs = self.getSecant(rhn)
        secant = coeffs[0]*range(self.x,rhn.x)+coeffs[1]
        data = data[self.x:rhn.x]
        delta = secant - data
        x = argmax(abs(delta))
        if x == 0 or x == len(data):
            x = -1
            y = -1
        else:
            y = data[x]
            x += self.x
            
        return x,y

class pois:
    def __init__(self,open,close,rate):
        self.close, self.open = close, open
        self.zeromean = close - open
        self.rate = rate

        self.points = list()
        
        self.points.append(poi(1,close[0]))
        x = argmax(close)
        self.addPoint(x,close[x])
        x = argmin(close)
        self.addPoint(x,close[x])
        x = len(close)-1
        self.addPoint(x,close[x])

    def addPoint(self,x,y):
        x+=1
        if x < self.points[0].x:
            self.points.insert(0,poi(x,y))
            return True
        elif x > self.points[-1].x:
            self.points.append(poi(x,y))
            return True

        for i in range(len(self.points)-1):
            if x > self.points[i].x and x < self.points[i+1].x:
                self.points.insert(i+1,poi(x,y))
                return True
            elif x == self.points[i].x or x == self.points[i+1].x:
                return False

    def addPoints(self):
        i = 0
        while i < len(self.points)-1:
            if self.checkStop(self.points[i],self.points[i+1],self.rate):
                i += 1
            else:
                x,y = self.points[i].getMidPoint(self.points[i+1],self.close)

                if x < 0 or not self.addPoint(x,y):
                    i += 1
                    continue

        self.tobin()
        return self

    def tobin(self):
        binrep = zeros(len(self.close))
        for i in range(len(self.points)-1):
            if self.points[i].y < self.points[i+1].y:
                binrep[self.points[i].x:self.points[i+1].x] = 1

        self.binrep = binrep
        return binrep

    def checkStop(self,p1,p2,window):
        stop = False
        if abs(p2.x - p1.x) < window:
            stop = True

        return stop

    def plotLines(self):
        '''
        scikit?
        matplotlib?
        '''

    def getDelta(self):
        delta = 0
        for i in range(len(self.points)-1):
            delta += abs(self.points[i+1].y - self.points[i].y)

        return delta

class interface:
    def __init__(self, api_key="", api_secret="", windowSize = 360, predLen = 5):
        self.pair = "BTCUSDT"
        self.api_key = api_key
        self.api_secret = api_secret
        self.windowSize = windowSize
        self.client = Client(api_key,api_secret)
        self.predLen = predLen
        self.model = None
        self.state = None

    def getCandles(self):
        return self.client.get_historical_klines(
            self.pair,
            self.client.KLINE_INTERVAL_1MINUTE,
            str(self.windowSize) + " min ago UTC"
            )

    def loadModel(self,path):
        self.model = load_model(path)

    def scaleData(self,open,close):
        data = zeros((self.windowSize))
        for i in range(self.windowSize):
            data[i] = close[i] - open[i]
        
        data = data-min(data)
        data = data/max(data)
        data = 2*data - 1

        return reshape(data,(1,self.windowSize,1))

    def getData(self):
        candles = self.getCandles()

        open = zeros((self.windowSize))
        close = zeros((self.windowSize))

        for i in range(self.windowSize):
            # 0:timestamp 1:open 2:high 3:low 4:close
            open[i] = float(candles[i][1])
            close[i] = float(candles[i][4])

        return open,close

    def fwp(self,open=None, close=None):
        if self.model is None:
            raise Exception("no prediction network")

        if open is None or close is None:
            open, close = self.getData()        

        prediction = self.model.predict(self.scaleData(open,close))
        return reshape(prediction,(self.windowSize))
    
    def iter(self,open=None,close=None):
        prediction = self.fwp(open,close)

        if self.state != 0 and prediction[-self.predLen-1] < 0.5:
            return self.short(close)
        elif self.state != 1 and prediction[-self.predLen-1] > 0.5:
            return self.long(close)
        else: 
            return None,0
    
    def long(self,data=None):
        '''
            create long request with binance api
        '''

        self.state = 1
        if data is not None:
            return True, data[-1]
        else: 
            # long request response from binance?
            return True

    def short(self,data=None):
        '''
            create short request with binance api
        '''

        self.state = 0
        if data is not None:
            return False, data[-1]
        else:
            # short request response from binance?
            return False