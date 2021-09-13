classdef layer_normalization_layer
    properties
        
        name
        
        offset
        scale_factor
    end
    
    methods
        function [obj,weights] = layer_normalization_layer(no_channels)
            weights.offset = gpudl(rand(no_channels,1),'C');
            weights.scale_factor = gpudl(rand(no_channels,1),'C');
        end
    end
    
    methods (Static)
        function x = fw(x,weights)
            x = layernorm(x,weights.offset,weights.scale_factor);
        end
    end  
 
end