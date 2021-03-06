from numpy import argmax,argmin,zeros

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
        self.zeromean = close - open
        self.data = close
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
                x,y = self.points[i].getMidPoint(self.points[i+1],self.data)

                if x < 0 or not self.addPoint(x,y):
                    i += 1
                    continue

        self.tobin()
        return self

    def tobin(self):
        binrep = zeros(len(self.data))
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


        
    
        
        
        
        
        
                
