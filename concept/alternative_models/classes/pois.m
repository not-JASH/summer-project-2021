classdef pois
    properties
        zeromean
        points
        rate
        data
        bin
    end
    
    methods 
        function obj = pois(data,rate)
            %close - open
            obj.zeromean = data(:,1)-data(:,2);
            data = data(:,1);
            obj.data = data;
            obj.rate = rate;
            obj.points = poi(1,data(1));
            [~,x] = min(data);
            obj = obj.addPoint(x,data(x,1));
            [~,x] = max(data);
            obj = obj.addPoint(x,data(x,1));
            obj = obj.addPoint(size(data,1),data(end,1));
        end
        
        function obj = addPoint(obj,x,y)
            if x < obj.points(1).x
                obj.points = [poi(x,y);obj.points];
                return
            elseif x > obj.points(end).x
                obj.points = [obj.points;poi(x,y)];
                return
            end
            
            for i = 1:size(obj.points,1)
                if x == obj.points(i).x || x == obj.points(i+1).x
                    return
                elseif x > obj.points(i).x && x < obj.points(i+1).x
                    obj.points = [obj.points(1:i);poi(x,y);obj.points(i+1:end)];
                end
            end        
        end
        
        function obj = addPoints(obj)
           i = 1;
           while (i < size(obj.points,1))
               if obj.checkStop(obj.points(i),obj.points(i+1),obj.rate)
                   i = i +1;
               else 
                   [x,y] = obj.points(i).getMidPoint(obj.points(i+1),obj.data);
                   
                   if x < 0
                       i = i+1;
                       continue;
                   end
                   
                   obj = obj.addPoint(x,y);
               end
           end 
           obj = obj.tobin;
        end
        
        function [obj,bin] = tobin(obj)
           bin = zeros(size(obj.data));
           for i = 1:size(obj.points,1)-1
               if obj.points(i).y < obj.points(i+1).y
                   bin(obj.points(i).x:obj.points(i+1).x) = 1;
               end
           end     
           obj.bin = bin;
        end
        
        function delta = getDelta(obj)
           delta = zeros(size(obj.points,1)-1,1);
           for i = 1:size(delta,1)
               delta(i) = abs(obj.points(i+1).y - obj.points(i).y);
           end
        end
        
        
        function stop = checkStop(obj,p1,p2,window)
            stop = false;
            if abs(p2.x - p1.x) < window
                stop = true;
            end           
        end
        
        function plotLines(obj)
           hold on
           for i = 1:size(obj.points,1)-1
               line([obj.points(i).x obj.points(i+1).x],[obj.points(i).y obj.points(i+1).y]);
           end    
           hold off            
        end
    end
end