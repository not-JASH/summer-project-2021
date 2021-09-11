classdef layer_normalization_layer
    properties
        
    end
    
    methods
        function obj = layer_normalization_layer
        end
        
        function x = fw(obj,x,varargin)
            x = layernorm(x,varargin{:});
        end
    end
    
end