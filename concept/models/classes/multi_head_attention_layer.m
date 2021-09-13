classdef multi_head_attention_layer
    
    properties
        
        name
        
        depth
        no_heads
        d_model
        
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
        function [obj,weights] = multi_head_attention_layer(input_channels,d_model,no_heads,causal,dropout)
            persistent count
            if isempty(count)
                count = 0;
            end
            count = count+1;
            
            assert(rem(d_model,no_heads) == 0,"d_model must be a multiple of no_heads");
            obj.depth =  floor(d_model/no_heads);
            obj.no_heads = no_heads;
            obj.d_model = d_model;
            
            [obj.w_query,weights.w_query] = fully_connected_layer(d_model,input_channels,"w_query_"+num2str(count));
            %obj.split_reshape_query = reshape_layer([],no_heads,depth,"split_reshape_query_"+num2str(count));
            %obj.split_permute_query = permute_layer([2 1 3],"split_permute_query_"+num2str(count));
            
            [obj.w_value,weights.w_value] = fully_connected_layer(d_model,input_channels,"w_value_"+num2str(count));
            %obj.split_reshape_value = reshape_layer([],no_heads,depth,"split_reshape_value_"+num2str(count));
            %obj.split_permute_value = permute_layer([2 1 3],"split_permute_value_"+num2str(count));
            
            [obj.w_key,weights.w_key] = fully_connected_layer(d_model,input_channels,"w_key_"+num2str(count));
            %obj.split_reshape_key = reshape_layer([],no_heads,depth,"split_reshape_key_"+num2str(count));
            %obj.split_permute_key = permute_layer([2 1 3],"split_permute_key_"+num2str(count));
            
            obj.attention = attention_layer(causal,dropout);
            %obj.join_permute_attention = permute_layer([2 1 3],"join_permute_attention_"+num2str(count));
            %obj.join_reshape_attention = reshape_layer([],d_model,1,"join_reshape_attention_"+num2str(count));
            
            [obj.dense,weights.dense] = fully_connected_layer(d_model,input_channels,"dense_"+num2str(count));    
        end
        
        function x = fw(obj,inputs,mask,weights)
            %make sure batch size is 3rd (4th?) dimension
            %dl array formats inputs such that it is 'CBT'
            %this means that before arrays are reshaped into parallel
            %attention layers, inputs must be rearranged to 'CTB'            
            
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
                        
            query = obj.w_query.fw(q,weights.w_query); %output dimensions at this step are expected to be 'CBT'
            query = rearrange(query);
            query = reshape(query,[],obj.no_heads,obj.depth,batch_size);    % split_reshape_query
            query = permute(query,[2 1 3 4]);                               % split_permute_query
            
            
            value = obj.w_value.fw(v,weights.w_value);
            value = rearrange(value);
            value = reshape(value,[],obj.no_heads,obj.depth,batch_size);    %split_reshape_value
            value = permute(value,[2 1 3 4]);                               %split_permute_value
            
            key = obj.w_key.fw(k,weights.w_key);
            key = rearrange(key);
            key = reshape(key,[],obj.no_heads,obj.depth,batch_size);        %split_reshape_key
            key = permute(key,[2 1 3 4]);                                   %split_permute_key
            
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
            x_attention = permute(x_attention,[2 1 3 4]);       %join_permute_attention
            x_attention = reshape(x_attention,obj.d_model,batch_size,[]);    %join_reshape_attention
            
            x = obj.dense.fw(x_attention,weights.dense,'dataformat','CTB');
            
            function data = rearrange(data)
                data = reshape(data,size(data)); %remove labels
                data = permute(data,[1 3 2]);                        
            end
        end
    end
end