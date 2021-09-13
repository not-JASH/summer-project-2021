classdef decoder_layer
    properties
        name
        
        multi_head_attention1
        dropout_attention1
        %add_attention1
        layer_norm_attention1
        
        multi_head_attention2
        dropout_attention2
        %add_attention2
        layer_norm_attention2
        
        dense1
        dense2
        dropout_dense
        %add_dense
        layer_norm_dense
        
    end
    
    methods
        function [obj,weights] = decoder_layer(input_channels,d_model,no_heads,dff,dropout)
            
            persistent count
            if isempty(count)
                count = 0;
            end
            count = count +1;
            
            [obj.multi_head_attention1,weights.multi_head_attention1] = ...
                multi_head_attention_layer(input_channels,d_model,no_heads,false,dropout);
            obj.dropout_attention1 = dropout_layer;
            %obj.add_attention1 = addition_layer;
            [obj.layer_norm_attention1,weights.layer_norm_attention1] = ...
                layer_normalization_layer(d_model);
            
            [obj.multi_head_attention2,weights.multi_head_attention2] = ...
                multi_head_attention_layer(input_channels,d_model,no_heads,false,dropout);
            obj.dropout_attention2 = dropout_layer;
            %obj.add_attention2 = addition_layer;
            [obj.layer_norm_attention2,weights.layer_norm_attention2] = ...
                layer_normalization_layer(d_model);
            
            [obj.dense1,weights.dense1] = fully_connected_layer(dff,d_model,"dense1_"+num2str(count));
            [obj.dense2,weights.dense2] = fully_connected_layer(d_model,dff,"dense2_"+num2str(count));
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
            
            %   multi_head_attention
            attention = obj.multi_head_attention1.fw({inputs{1},inputs{1},inputs{1}},mask,weights.multi_head_attention1);
            attention = obj.dropout_attention1.fw(attention,dropout_rate);
            x = inputs{1} + attention;
            x = obj.layer_norm_attention1.fw(x,weights.layer_norm_attention1);
            
            attention = obj.multi_head_attention2.fw({x,inputs{1},inputs{1}},mask,weights.multi_head_attention2);
            attention = obj.dropout_attention2.fw(attention,dropout_rate);
            x = x + attention;
            x = obj.layer_norm_attention2.fw(x,weights.layer_norm_attention2);
            
            % feed forward
            dense = obj.dense1.fw(x,weights.dense1);
            dense = obj.dense2.fw(dense,weights.dense2);
            dense = obj.dropout_dense.fw(dense,dropout_rate);
            x = x + dense;
            x = obj.layer_norm_dense.fw(x,weights.layer_norm_dense);            
            
        end
  
    end
end