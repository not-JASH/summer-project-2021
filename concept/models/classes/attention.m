classdef attention 
    properties 
        causal
        dropout        
    end
        
    methods 
        function obj = attention(causal,dropout)
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
            
            batch_size = size(q,4);
            assert(batch_size == size(v,4) && batch_size == size(k,4),...
                "inconsistent batch sizes between inputs");
            
            dk = size(q,2); %which dimension is the key dimension?
            
            q = permute(q,[2 1 3 4]);
            v = permute(v,[2 1 3 4]);
            k = permute(k,[2 1 3 4]);
            
            attention = pagemtimes(q,permute(k,[2 1 3 4]));
            attention = softmax(attention/sqrt(dk));
            attention = pagemtimes(attention,v);
            
            attention = permute(attention,[2 1 3 4]);
        end
    end
end