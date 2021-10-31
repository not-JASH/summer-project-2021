classdef reshape_layer < nnet.layer.Layer
    properties
        nd1
        nd2
        nd3
    end
    
    methods 
        function layer = reshape_layer(nd1,nd2,nd3,name)
            layer.nd1 = nd1;
            layer.nd2 = nd2;
            layer.nd3 = nd3;
            layer.name = name;
            %layer.description = ...
        end
        
        function x = fw(layer,x)
            x = reshape(x,layer.nd1,layer.nd2,layer.nd3);
        end
    end
end
