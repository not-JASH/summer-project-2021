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
        function obj = encoder_layer(input_channels,d_model,no_heads,dff,dropout)
            
            persistent count
            if isempty(count) 
                count = 0;
            end
            count = count+1;
            
            obj.multi_head_attention = ...
               multi_head_attention_layer(input_channels,d_model,no_heads,false,dropout); 
            obj.dropout_attention = dropout_layer; 
            %obj.add_attention = addition_layer;
            obj.layer_norm_attention = layer_normalizationlayer;
           
            obj.dense1 = fully_connected_layer(dff,d_model,'dense1_'+num2str(count));
            obj.dense2 = fully_connected_layer(d_model,dff,'dense2_'+num2str(count));
            obj.dropout_dense = dropout_layer;
            %obj.add_dense = addition_layer;
            obj.layer_norm_dense = layer_normalization_layer;
        end
        
        function x = fw(obj,inputs,mask)
            
            %{
                9/10/2021 
                - Dropout layer not implemented
                - Not sure if fully connected layers are the right size.           
            
            %}
            
            
            
            dropout_rate = 0.3;
            
            %   multihead attention
            attention = obj.multi_head_attention.fw({inputs,inputs,inputs},mask);
            attention = obj.dropout_layer.fw(attention,dropout_rate);   %
            x = attention + inputs;
            x = obj.layer_norm_attention.fw(x);
            
            %   feed forward
            dense = obj.dense1.fw(x);
            dense = obj.dense2.fw(dense);
            dense = obj.dropout_dense.fw(dense,dropout_rate);           
            x = x + dense;
            x = obj.layer_norm_dense.fw(x);
        end
    end
end