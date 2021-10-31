classdef attention_layer
    properties 
        causal
        dropout        
    end
        
    methods 
        function obj = attention_layer(causal,dropout)
            obj.causal = causal;
            %layer.dropout = dropoutlayer(dropout)
            
        end
        
        function attention = fw(obj,inputs,mask,d_model)
                        
            q = inputs{1};
            v = inputs{2};
            if length(inputs)>2
                k = inputs{3};
            else
                k = v;
            end
            
            batch_size = size(q,2);
            assert(batch_size == size(v,2) && batch_size == size(k,2),...
                "inconsistent batch sizes between inputs");
            
            dk = size(q,3); %which dimension is the key dimension?
            
            % [ depth batch_size channels timesteps] 
            % => [channels timesteps depth batch_size]
            
            
            q = permute(q,[3 4 1 2]);
            v = permute(v,[3 4 1 2]);
            k = permute(k,[3 4 1 2]);
                        
            attention = pagemtimes(q,permute(k,[2 1 3 4]));
            attention = softmax(attention/sqrt(dk),'dataformat','CTSB');
            attention = pagemtimes(attention,v);
            
            attention = permute(attention,[2 1 3 4]);
        end
    end
end