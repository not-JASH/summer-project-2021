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
        confidence_interval
        heuristic_limit
        last_prediction
    end
    
    methods
        function obj = interface(window_size,prediction_length,offset,confidence_interval,heuristic_limit)  
            obj.pair = "BTCUSDT";
            obj.window_size = window_size;
            obj.prediction_length = prediction_length;     
            obj.state = -1;
            obj.offset = offset;
            obj.s_counter = 0;
            obj.l_counter = 0;
            obj.confidence_interval = confidence_interval;
            obj.heuristic_limit = heuristic_limit;
            obj.last_prediction = gpuArray(dlarray(rand(1,window_size),'CT'));
        end
        
        function [obj,prediction] = fwp(obj,sample)
           sample = scale_data(sample);
%            try 
%                prediction = predict(obj.model,gpuArray(dlarray(sample,'CT')),obj.last_prediction);
%                obj.last_prediction = prediction;
%                return
%            catch err
%                disp(err);
%            end          
           
           try
               [obj.model,prediction] = predictAndUpdateState(obj.model,sample);
           catch
               prediction = predict(obj.model,sample);
           end
        end
        
        function [obj,state,price,prediction] = iter(obj,sample,close)
            
            [obj,prediction] = obj.fwp(sample);            
            loc = length(prediction)-obj.prediction_length+obj.offset;
            
            if obj.state ~= 0 && prediction(loc) < obj.confidence_interval
            %if obj.state ~= 0 && prediction(1) > 1 - confidence_interval
                obj.l_counter = 0;
                if obj.s_counter == obj.heuristic_limit
                    [state,price] = obj.short(close);
                    obj.state = 0;
                    obj.s_counter = 0;
                else
                    [state,price] = deal([],-1);
                    obj.s_counter = obj.s_counter+1;
                end
            elseif obj.state ~= 1 && prediction(loc) > 1 - obj.confidence_interval
            %elseif obj.state ~= 1 && prediction(2) > 1 - confidence_interval    
                obj.s_counter = 0;
                if obj.l_counter == obj.heuristic_limit
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