from binance.client import Client
from datetime import timedelta
from numpy import savetxt


api_key     = ""
api_secret  = ""

client=Client(api_key,api_secret)
klines=client.get_historical_klines("BTCUSDT",client.KLINE_INTERVAL_5MINUTE,"January,01,2021","September,20,2021")
textname = "BTCUSDT_5M.txt"
savetxt("./"+textname,klines,delimiter=" ",fmt="%s")

