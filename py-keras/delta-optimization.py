from numpy import argmax,argmin

class poi:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getSecant(self,rhn):
        m = (rhn.y - self.y)/(rhn.x - self.x)
        b = self.y-(b*self.x);
        return [m,x]

    def getMidPoint(self,rhn,data):
        coeffs = self.getSecant(rhn)
        secant = coeffs[0]*range(self.x,rhn.x)+coeffs[1]
        data = data(range(self.x,rhn.x))
        delta = secant - data
        x = argmax(abs(delta))
        
        if x == 1 || x == len(data):
            x = -1
            y = -1
        else:
            y = data[x]
            x += self.x
            
        return x,y

class pois:
    def __init__(self,data,rate):
        self.zeromean = data[0]-data[1]
        data = data[0]
        self.data = data
        self.rate = rate
        self.points = poi(1,data[0])
        x = argmax(data)
        self.addPoint(x,data[x])
        x = argmin(data)
        self.addPoint(x,data[x])
        x = len(data)-1
        self.addPoint(x,data[x])

    def addPoint(self,x,y):
        if x < self.points[0].x:
            self.points.insert(0,poi(x,y))
            return
        else if x > self.points[-1].x:
            self.points.append(poi(x,y))
            return

        
        
        
        
        
                
