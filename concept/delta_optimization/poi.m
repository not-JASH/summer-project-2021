classdef poi 
    properties 
        x
        y   
    end
    
    methods
        function obj = poi(x,y)
            obj.x = x;
            obj.y = y;
        end
        
        function [x,y] = getMidPoint(obj,rhn,data)

           coeffs = polyfit([obj.x, rhn.x],[obj.y,rhn.y],1);
           % secant = coeffs(1)*x + coeffs(2);
           secant = coeffs(1)*([obj.x:rhn.x]) + coeffs(2);
           data = data(obj.x:rhn.x);
           delta = secant'-data;
           
           [~,x] = max(abs(delta));
           if x == 1 || x == size(data,1)
               x = -1;
               y = -1;
               return
           end
           y = data(x);
           x = x + obj.x-1;          
        end       
        
    end
end

