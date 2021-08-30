from tensorflow import keras
from keras.models import load_model
from binance.client import Client
from numpy import zeros,reshape


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