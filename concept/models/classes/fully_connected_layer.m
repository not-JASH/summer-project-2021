classdef fully_connected_layer
    properties
        name
    end
    
    methods
        function [obj,weights] = fully_connected_layer(output_features,input_channels,name)
            obj.name = name;
            
            weights.weights = gpudl(rand(output_features,input_channels),'');
            weights.bias = gpudl(rand(output_features,1),'');
        end 
    end
    
    methods (Static)
        function y = fw(x,weights,varargin)
           y = fullyconnect(x,weights.weights,weights.bias,varargin{:});
        end
    end
end