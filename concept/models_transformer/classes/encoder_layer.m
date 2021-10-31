classdef encoder_layer
    properties
        multi_head_attention
        dropout_attention
        %add_attention
        layer_norm_attention
        
        dense1
        dense2
        dropout_dense
        %add_dense
        layer_norm_dense        
    end
    
    methods
        function [obj,weights] = encoder_layer(input_channels,d_model,no_heads,dff,dropout_rate)
            
            persistent count
            if isempty(count) 
                count = 0;
            end
            count = count+1;
            
            [obj.multi_head_attention,weights.multi_head_attention] = ...
               multi_head_attention_layer(input_channels,d_model,no_heads,false,dropout_rate); 
            obj.dropout_attention = dropout_layer; 
            %obj.add_attention = addition_layer;
            [obj.layer_norm_attention,weights.layer_norm_attention] = ...
                layer_normalization_layer(d_model);
           
            [obj.dense1, weights.dense1] = fully_connected_layer(dff,d_model,'dense1_'+num2str(count));
            [obj.dense2, weights.dense2] = fully_connected_layer(d_model,dff,'dense2_'+num2str(count));
            obj.dropout_dense = dropout_layer;
            %obj.add_dense = addition_layer;
            [obj.layer_norm_dense,weights.layer_norm_dense] = layer_normalization_layer(d_model);
        end
        
        function x = fw(obj,inputs,mask,dropout_rate,weights)
            
            %{
                9/10/2021 
                - Dropout layer not implemented
                - Not sure if fully connected layers are the right size.           
            
            %}
            
            
            %   multihead attention
            attention = obj.multi_head_attention.fw({inputs,inputs,inputs},mask,weights.multi_head_attention);
            attention = obj.dropout_attention.fw(attention,dropout_rate);   
            
            x = attention + inputs;
            x = obj.layer_norm_attention.fw(x,weights.layer_norm_attention);
            
            %   feed forward
            dense = obj.dense1.fw(x,weights.dense1);
            dense = obj.dense2.fw(dense,weights.dense2);
            dense = obj.dropout_dense.fw(dense,dropout_rate);           
            x = x + dense;
            x = obj.layer_norm_dense.fw(x,weights.layer_norm_dense);
        end
    end
end