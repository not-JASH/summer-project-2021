classdef dropout_layer
    properties
        
    end
    
    methods
        function obj = dropout_layer
        end
    end
    
    methods (Static)
        function data = fw(data,rate)
            %   nice 
            data(normrnd(0,0.1,size(data)) < norminv(rate,0,0.1)) = 0;
        end
    end
end