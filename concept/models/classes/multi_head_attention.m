classdef multi_head_attention 
    
    properties
        
        depth
        
        w_query
        %split_reshape_query
        %split_permute_query       
        
        w_value
        %split_reshape_value
        %split_permute_value
           
        w_key
        %split_reshape_key
        %split_permute_key
        
        attention
        %join_permute_attention
        %join_reshape_attention
        
        dense
    end
    
    methods 
        function obj = multi_head_attention(input_channels,d_model,no_heads,causal,dropout)
            persistent count
            if isempty(count)
                count = 0;
            end
            count = count+1;
            
            assert(rem(d_model,no_heads) == 0,"d_model must be a multiple of no_heads");
            obj.depth =  floor(d_model/no_heads);
            
            obj.w_query = fully_connected_layer(d_model,input_channels,"w_query_"+num2str(count));
            %obj.split_reshape_query = reshape_layer([],no_heads,depth,"split_reshape_query_"+num2str(count));
            %obj.split_permute_query = permute_layer([2 1 3],"split_permute_query_"+num2str(count));
            
            obj.w_value = fully_connected_layer(d_model,input_channels,"w_value_"+num2str(count));
            %obj.split_reshape_value = reshape_layer([],no_heads,depth,"split_reshape_value_"+num2str(count));
            %obj.split_permute_value = permute_layer([2 1 3],"split_permute_value_"+num2str(count));
            
            obj.w_key = fullyConnectedLayer(d_model,'Name',"w_key_"+num2str(count));
            %obj.split_reshape_key = reshape_layer([],no_heads,depth,"split_reshape_key_"+num2str(count));
            %obj.split_permute_key = permute_layer([2 1 3],"split_permute_key_"+num2str(count));
            
            obj.attention = attention(causal,dropout);
            %obj.join_permute_attention = permute_layer([2 1 3],"join_permute_attention_"+num2str(count));
            %obj.join_reshape_attention = reshape_layer([],d_model,1,"join_reshape_attention_"+num2str(count));
            
            obj.dense = fully_connected_layer(d_model,input_channels,'Name',"dense_"+num2str(count));    
        end
        
        function x = fw(obj,inputs,mask)
            %make sure batch size is 3rd (4th?) dimension
            
            q = inputs{1};
            v = inputs{2};
            if length(inputs)>2
                k = inputs{3};
            else
                k = v;
            end
            
            batch_size = size(q,3);
            assert(batch_size == size(v,3) && batch_size == size(k,3),...
                "inconsistent batch sizes between inputs");
            
            query = obj.w_query.fw(q);
            query = reshape(query,[],no_heads,obj.depth,batch_size);    % split_reshape_query
            query = permute(query,[2 1 3 4]);                           % split_permute_query
            
            value = obj.w_value.fw(v);
            value = reshape(value,[],no_heads,obj.depth,batch_size);    %split_reshape_value
            value = permute(value,[2 1 3 4]);                           %split_permute_value
            
            key = obj.w_key.fw(k);
            key = reshape(key,[],no_heads,obj.depth,batch_size);        %split_reshape_key
            key = permute(key,[2 1 3 4]);                               %split_permute_key
            
%             if ~isempty(mask)
%                 if ~isempty(mask{1})
%                     mask{1} = reshape(mask{1},[],1);
%                     mask{1} = permute(mask{1},[2 1]);
%                 end
%                 
%                 if ~isempty(mask{2})
%                     mask{2} = reshape(mask{2},[],1);
%                     mask{2} = permute(mask{2},[2 1]);
%                 end
%             end
            
            x_attention = obj.attention.fw({query, value, key},mask);
            x_attention = permute(x_attention,[2 1 3]);         %join_permute_attention
            x_attention = reshape(x_attention,[],d_model,1);    %join_reshape_attention
                       
            x = obj.dense.fw(x_attention);
        end
        
        function [] = forward(layer)
            
        end
        
        function [] = backward(layer)
            
        end            
    end
end