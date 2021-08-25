from binance.client import Client
from datetime import timedelta
from numpy import savetxt


api_key     = ""
api_secret  = ""

client=Client(api_key,api_secret)
klines=client.get_historical_klines("BTCUSDT",client.KLINE_INTERVAL_1MINUTE,"January,01,2021","August,14,2021")
textname = "BTCUSDT.txt"
savetxt("./"+textname,klines,delimiter=" ",fmt="%s")
