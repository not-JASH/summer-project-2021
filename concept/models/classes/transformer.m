classdef transformer
    
    properties
        
        %   global properties
        d_model
        
        %   encoder
        
        encoder_embedding
        encoder_pos
        encoder_layers
        encoder_dropout
        
        % decoder      
        
        decoder_embedding
        decoder_pos
        decoder_layers
        decoder_dropout
        
        % dense (fully connected)
        dense
        
    end
    
    methods
        function [obj,weights] = transformer(input_channels,target_size,no_layers,d_model,no_heads,dff,dropout_rate)

            
            %{
                9/10/2021
                
                stores positional_encoding as an array in object, does this
                count as a learnable parameter? 
            
                is the dimensionality of positional_encoding correct? does
                it act along the correct dimensions          
            %}
            
            
            maximum_position_encoding = 1e4;
            
            obj.d_model = d_model;
            
            
            %   initialize encoder weights & layers
            
            [obj.encoder_embedding,weights.encoder_embedding] = ...
                fully_connected_layer(d_model,input_channels,"encoder_embedding");
            obj.encoder_pos = obj.positional_encoding(maximum_position_encoding,d_model);
            obj.encoder_pos = permute(obj.encoder_pos,[3 1 2]);
            
            for i = 1:no_layers
                [obj.encoder_layers{i},weights.encoder_layers{i}] = ...
                    encoder_layer(d_model,d_model,no_heads,dff,dropout_rate);
            end
            
            obj.encoder_dropout = dropout_layer;  
            
            
            %   initialize decoder weights & layers
            
            [obj.decoder_embedding,weights.decoder_embedding] = ...
                fully_connected_layer(d_model,input_channels,"decoder_embedding");
            obj.decoder_pos = obj.positional_encoding(maximum_position_encoding,d_model);
            obj.decoder_pos = permute(obj.decoder_pos,[3 1 2]);
            
            for i = 1:no_layers
                [obj.decoder_layers{i},weights.decoder_layers{i}] = ...
                    decoder_layer(d_model,d_model,no_heads,dff,dropout_rate);
            end
            
            obj.decoder_dropout = dropout_layer;
            
            [obj.dense,weights.dense] = fully_connected_layer(target_size,d_model,"transformer_dense");            
        end
       
        function x = encoder_fw(obj,inputs,dropout_rate,weights)

            %{
            
                9/10/2021
                
                what is and how to calculate an embedding mask
                what happens if inputs is a cell array 
            
            %}           
            
            x = obj.encoder_embedding.fw(inputs,weights.encoder_embedding);

            % positional encoding
            x = x*sqrt(obj.d_model);
            x = x+obj.encoder_pos(:,:,1:size(x,3)); % is this the correct dimension
            
            x = obj.encoder_dropout.fw(x,dropout_rate);
            
            % encoder layer
            %embedding_mask = obj.encoder_embedding.compute_mask(inputs);
            embedding_mask = [];            

            for i = 1:length(obj.encoder_layers)
                x = obj.encoder_layers{i}.fw(x,embedding_mask,dropout_rate,weights.encoder_layers{i});
            end
            
        end
        
        function x = decoder_fw(obj,inputs,dropout_rate,weights)
            
            %{
                9/10/2021
                
                how to calcualte an embedding mask               
                
            
            %}
      
            x = obj.decoder_embedding.fw(inputs{1},weights.decoder_embedding);
            
            % positional encoding
            x = x*sqrt(obj.d_model);
            x = x+obj.decoder_pos(:,:,1:size(x,3));
            
            x = obj.decoder_dropout.fw(x,dropout_rate);
            
            % decoder layers
            %embedding_mask = obj.decoder_embedding.compute_mask(inputs{1})
            embedding_mask = [];
            for i = 1:length(obj.decoder_layers)
                x = obj.decoder_layers{i}.fw({x,inputs{2}},embedding_mask,dropout_rate,weights.decoder_layers{i});
            end
        end
        
        function y = fw(obj,inputs,targets,dropout_rate,weights)
            
            y = obj.encoder_fw(inputs,dropout_rate,weights);
            
            y = obj.decoder_fw({targets,y},dropout_rate,weights);
            
            y = obj.dense.fw(y,weights.dense);            
        end
   
        function pos_encoding = positional_encoding(obj,position,d_model)
            angle_rads = obj.get_angles([1:position]',[1:d_model],d_model);

            % apply sin to even indicies in array 
            angle_rads(:,2:2:end) = sin(angle_rads(:,2:2:end));
            
            % apply cos to odd indicies in array
            angle_rads(:,1:2:end) = cos(angle_rads(:,1:2:end));
            
            pos_encoding = reshape(angle_rads,[1 size(angle_rads)]);          
        end
        
        function angle_rates = get_angles(obj,pos,i,d_model)
            angle_rates = 1./power(10000,(2*(i/2)/d_model));
            
            angle_rates = angle_rates.*pos;
        end
    end
end