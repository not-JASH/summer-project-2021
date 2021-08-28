'''
    binance api is in html/node and python
  
'''

from classes import interface
from os import getcwd


sample = interface()

path = 'version1'
#path = getcwd() + '/implementation/trade-interface/' + path
sample.loadModel(path)


test = sample.iter()
