classdef interface
    properties 
        pair
        window_size
        prediction_length
        model
        state        
        offset
    end
    
    methods
        function obj = interface(window_size,prediction_length,offset)  
            obj.pair = "BTCUSDT";
            obj.window_size = window_size;
            obj.prediction_length = prediction_length;     
            obj.state = -1;
            obj.offset = offset;
        end
        
        function prediction = fwp(obj,sample)
           sample = scale_data(sample);
           prediction = predict(obj.model,sample);            
        end
        
        function [state,price] = iter(obj,sample,close)
            prediction = obj.fwp(sample);
            loc = length(prediction)-obj.prediction_length+obj.offset;
            if obj.state ~= 0 && prediction(loc) < 0.5
                [state,price] = obj.short(close);
            elseif obj.state ~= 1 && prediction(loc) > 0.5
                [state,price] = obj.long(close);
            else 
                state = [];
                price = -1;
            end
        end
        
        function [state,price] = long(obj,close)
            state = true;
            price = close(end);            
        end
        
        function [state,price] = short(obj,close)
            state = false;
            price = close(end);
        end
    end
end