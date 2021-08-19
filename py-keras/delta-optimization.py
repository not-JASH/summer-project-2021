from numpy import argmax,argmin,zeros

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

        for i in range(len(self.points)-1):
            if x > self.points[i].x && x < self.points[i+1].x:
                self.points.insert(i+1,poi(x,y))

    def addPoints(self):
        i = 0
        while i < len(self.points)-1:
            if self.checkStop(self.points[i],self.points[i+1],self.rate):
                i += 1
            else:
                x,y = self.points[i].getMidPoint(points[i+1],self.data)

                if x < 0:
                    continue

                self.addPoint(x,y)

        self.tobin

    def tobin(self):
        binrep = zeros(self.points.shape)
        for i in range(len(points)-1):
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
            delta += abs(points[i+1].y - points[i].y)

        return delta
        
                
        
                
        
    
        
        
        
        
        
                
