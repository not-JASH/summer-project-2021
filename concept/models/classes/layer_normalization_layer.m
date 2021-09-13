classdef layer_normalization_layer
    properties
        
    end
    
    methods
        function obj = layer_normalization_layer
        end
    end
    
    methods (Static)
        function x = fw(x,varargin)
            x = layernorm(x,varargin{:});
        end
    end
    
end