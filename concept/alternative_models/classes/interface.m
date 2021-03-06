classdef interface
    properties 
        pair
        window_size
        prediction_length
        model
        state        
        offset
        s_counter
        l_counter
    end
    
    methods
        function obj = interface(window_size,prediction_length,offset)  
            obj.pair = "BTCUSDT";
            obj.window_size = window_size;
            obj.prediction_length = prediction_length;     
            obj.state = -1;
            obj.offset = offset;
            obj.s_counter = 0;
            obj.l_counter = 0;
        end
        
        function [obj,prediction] = fwp(obj,sample)
           sample = scale_data(sample);
           
           try
               [obj.model,prediction] = predictAndUpdateState(obj.model,sample);
           catch
               prediction = predict(obj.model,sample);
           end
        end
        
        function [obj,state,price,prediction] = iter(obj,sample,close)
            
            confidence_interval = 0.5;
            count_lim = 0;
            
            %prediction = obj.fwp(sample);
            [obj,prediction] = obj.fwp(sample);
            %loc = length(prediction)-obj.prediction_length+obj.offset;
            %if obj.state ~= 0 && prediction(loc) < confidence_interval
            if obj.state ~= 0 && prediction(1) > 1 - confidence_interval
                obj.l_counter = 0;
                if (obj.s_counter == count_lim)
                    [state,price] = obj.short(close);
                    obj.state = 0;
                    obj.s_counter = 0;
                else
                    [state,price] = deal([],-1);
                    obj.s_counter = obj.s_counter+1;
                end
            %elseif obj.state ~= 1 && prediction(loc) > 1 - confidence_interval
            elseif obj.state ~= 1 && prediction(2) > 1 - confidence_interval    
                obj.s_counter = 0;
                if obj.l_counter == count_lim
                    [state,price] = obj.long(close);
                    obj.state = 1;
                    obj.l_counter = 0;
                else
                    [state,price] = deal([],-1);
                    obj.l_counter = obj.l_counter + 1;
                end  
            else 
                [state,price] = deal([],-1);
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