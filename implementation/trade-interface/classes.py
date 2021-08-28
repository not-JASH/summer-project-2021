from tensorflow import keras
from keras.models import load_model
from binance.client import Client
from numpy import zeros


class interface:
    def __init__(self, api_key="", api_secret="", windowSize = 360):
        self.pair = "BTCUSDT"
        self.api_key = api_key
        self.api_secret = api_secret
        self.windowSize = windowSize
        self.client = Client(api_key,api_secret)
        self.model = None

    def getCandles(self):
        return self.client.get_historical_klines(
            self.pair,
            self.client.KLINE_INTERVAL_1MINUTE,
            str(self.windowSize) + " min ago UTC"
            )

    def loadModel(self,path):
        self.model = load_model(path)

    def iter(self):
        candles = self.getCandles()
        data = zeros((1,1,self.windowSize))
        for i in range(self.windowSize):
            # 0:timestamp 1:open 2:high 3:low 4:close
            data[0][0][i] = float(candles[i][4]) - float(candles[i][1])

        if self.model is None:
            raise Exception("load a model before iteration")

        print(data.shape)
        prediction = self.model.predict(data)
        print(prediction.shape)
        return prediction
        








