classdef fully_connected_layer
    properties
        name
        weights
        bias
    end
    
    methods
        function obj = fully_connected_layer(output_features,input_channels,name)
            obj.name = name;
            obj.weights = gpudl(rand(output_features,input_channels),'');
            obj.bias = gpudl(rand(output_features,1),'');
        end 
        
        function y = fw(obj,x)
           y = fullyconnect(x,obj.weights,obj.bias);
        end
    end
end