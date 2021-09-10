classdef permute_layer < nnet.layer.Layer
    properties
        new_order
    end
  
    methods 
        function layer = permute_layer(new_order,name)
            layer.new_order = new_order;
            layer.name = name;               
        end
        
        function x = fw(layer,x)
            x = premute(x,layer.new_order);
        end
    end
end