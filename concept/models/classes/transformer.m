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
        
    end
    
    methods
        function obj = transformer(input_channels,target_size,no_layers,d_model,no_heads,dff,dropout)
            
            obj.d_model = d_model;
            
            
            
        end
        
        function obj = init_encoder(obj,input_channels,no_layers,d_model,no_heads,dff,dropout)
            
            %{
                9/10/2021
                
                stores positional_encoding as an array in object, does this
                count as a learnable parameter? 
            
                is the dimensionality of positional_encoding correct? does
                it act along the correct dimensions          
            %}
            
            maximum_position_encoding = 1e4;
            
            obj.encoder_embedding = fully_connected_layer(d_model,input_channels,"encoder_embedding");
            obj.encoder_pos = obj.positional_encoding(maximum_position_encoding,d_model);
            
            for i = 1:no_layers
                obj.encoder_layers{i} = encoder_layer(input_channels,d_model,no_heads,dff,dropout);
            end
            
            obj.encoder_dropout = dropout_layer;         
        end
        
        function obj = init_decoder(obj,target_size,no_layers,d_model,no_heads,dff,dropout)
            
            %{
                9/10/2021
            
                same blurb as in init_decoder
            %}
            
            maximum_position_encoding = 1e4;
            
            obj.decoder_embedding = fully_connected_layer(target_size,d_model,"decoder_embedding");
            obj.decoder_pos = obj.positional_encoding(maximum_position_encoding,d_model);
            
            for i = 1:no_layers
                obj.decoder_layers{i} = decoder_layer(target_size,d_model,no_heads,dff,dropout);
            end
            
            obj.decoder_dropout = dropout_layer;
        end
        
        function x = encoder_fw(obj,inputs,dropout_rate)
            
            %{
            
                9/10/2021
                
                what is and how to calculate an embedding mask
                what happens if inputs is a cell array 
            
            %}           
                        
            x = obj.encoder_embedding.fw(inputs);
            
            % positional encoding
            x = x*sqrt(obj.d_model);
            x = x+obj.encoder_pos(:,size(x,2),:); % is this the correct dimension
            
            x = obj.encoder_dropout.fw(x,dropout_rate);
            
            % encoder layer
            %embedding_mask = obj.encoder_embedding.compute_mask(inputs);
            embedding_mask = [];            
            for i = 1:length(obj.encoder_layers)
                x = obj.encoder_layers{i}.fw(x,embedding_mask);
            end
            
        end
        
        function x = decoder_fw(obj,inputs,dropout_rate)
            
            %{
                9/10/2021
                
                how to calcualte an embedding mask               
                
            
            %}
            
            x = obj.decoder_embedding(inputs{1});
            
            % positional encoding
            x = x*sqrt(obj.d_model);
            x = x+obj.decoder_pos(:,size(shape,1),:);
            
            x = obj.decoder_dropout(x,dropout_rate);
            
            % decoder layers
            %embedding_mask = obj.decoder_embedding.compute_mask(inputs{1})
            embedding_mask = [];
            for i = 1:length(obj.decoder_layers)
                x = obj.decoder_layer{i}.fw({x,inputs{2}},embedding_mask);
            end
        end
        
        function y = fw(obj,x)
            
        end
        
        function pos_encoding = positional_encoding(obj,position,d_model)
            angle_rads = obj.get_angles([1:position],[1:position]',d_model);
            
            % apply sin to even indicies in array 
            angle_rads(:,2:2:position) = sin(angle_rads(:,2:2:position));
            
            % apply cos to odd indicies in array
            angle_rads(:,1:2:position) = cos(angle_rads(:,1:2:position));
            
            pos_encoding = reshape(angle_rads,[1 size(angle_rads)]);          
        end
        
        function angle_rates = get_angles(obj,pos,i,d_model)
            angle_rates = 1/power(10000,(2*(i/2)/d_model));
            angle_rates = angle_rates*pos;
        end
    end
end