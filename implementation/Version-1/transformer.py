from keras.layers import Layer, Embedding, Dropout, Add, LayerNormalization, Dense, Reshape, Permute, Attention
from tensorflow.math import sqrt
from tensorflow import cast, float32,shape



'''
    This transformer pulled and adapted from
    https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83

    d_model -> hidden_layers
'''

class Encoder(Layer):
    def __init__(self,input_size,no_layers=4,hidden_layers=512,no_heads=8,dff=2048,maximum_position_encoding=10000,dropout=0.0):
        super(Encoder,self).__init__()

        self.hidden_layers = hidden_layers
        self.embedding = Embedding(input_size,hidden_layers,mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding, hidden_layers)
        self.encoder_layers = [ EncoderLayer(hidden_layers=hidden_layers,no_heads=no_heads,dff=dff,dropout=dropout) for _ in range(no_layers)]
        self.dropout = Dropout(dropout)

    def call(self,inputs,mask=None,training=None):
        x = self.embedding(inputs)
        #positional encoding
        x *= sqrt(cast(self.hidden_layers,float32))
        x += self.pos[:,shape(x)[1],:]

        x = self.dropout(x, training=training)

        #Encoder Layer
        embedding_mask = self.embedding.compute_mask(inputs)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=embedding_mask)

        return x

    def compute_mask(self, inputs, mask=None):
        return self.embedding.compute_mask(inputs)

class Decoder(Layer):
    def __init__(self,target_size,no_layers=4,hidden_layers=512,no_heads=8,dff=2048,maximum_position_encoding=10000,dropout=0.0):
        super(Decoder,self).__init__()

        self.hidden_layers = hidden_layers
        self.embedding = Embedding(target_size,hidden_layers,mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding,hidden_layers)

        self.decoder_layers = [ DecoderLayer(hidden_layers=hidden_layers, no_heads=no_heads, dff=dff, dropout=dropout) for _ in range(no_layers)]

        self.dropout = Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        x = self.embedding(inputs[0])
        #positional encoding
        x *= sqrt(cast(hidden_layers,float32))
        x += self.pos[:,shape(x)[1],:]

        x = self.dropout(x, training=training)

        #decoder layer
        embedding_mask = self.embedding.compute_mask(inputs[0])
        for decoder_layer in self.decoder_layers:
            x = decoder_layer([x,inputs[1]], mask=[embedding_mask,mask])

        return x

    # comment this out to use masked_loss
    def compute_mask(self, inputs, mask=None):
        return self.embedding.compute_mask(inputs[0])

class EncoderLayer(Layer):
    def __init__(self,hidden_layers=512,no_heads=8,dff=2048,dropout=0.0):
        super(EncoderLayer,self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_layers,no_heads)
        self.dropout_attention = Dropout(dropout)
        self.add_attention = Add()
        self.layer_norm_attention = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(hidden_layers)
        self.dropout_dense = Dropout(dropout)
        self.add_attention = Add()
        self.layer_norm_dense = LayerNormalization(epsilon=1e-6)

    def call(self,inputs,mask=None,training=None):
        #print(mask)
        attention = self.multi_head_attention([inputs,inputs,inputs],mask = [mask,mask])
        attention = self.dropout_attention(attention,training=training)
        x = self.add_attention([inputs,attention])
        x = self.layer_norm_attention(x)
        #x = inputs

        ## Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense,training=training)
        x = self.add_dense([x,dense])
        x = self.layer_norm_dense

        return x

class DecoderLayer(Layer):
    def __init__(Self,hidden_layers=512,no_heads=8,dff=2048,dropout=0.0):
        super(DecoderLayer,self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(hidden_layers,no_heads,casual=True)
        self.dropout_attention1 = Dropout(dropout)
        self.add_attention1 = Add()
        self.layer_norm_attention1 = LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(hidden_layers,no_heads)
        self.dropout_attention2 = Dropout(dropout)
        self.add_attention2 = Add()
        self.layer_norm_attention2 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(dff,activation='relu')
        self.dense2 = Dense(hidden_layers)
        self.dropout_dense = Dropout(dropout)
        self.add_dense = Add()
        self.layer_norm_dense = LayerNormalization(epsilon=1e-6)

    def call(self,inputs,mask=None,training=None):
        # print(mask)
        attention = self.multi_head_attention1([inputs[0],inputs[0],inputs[0]], mask = [mask[0],mask[1]])
        attention = self.dropout_attention2(attention,training=training)
        x = self.add_attention([x,attention])
        x = self.layer_norm_attention1(x)

        ## Feed  Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense([x, dense])
        x = self.add_dense([x,dense])
        x = self.layer_norm_dense(x)
        
        return x

class MultiHeadAttention(Layer):
    def __init__(Self,hidden_layers=512,no_heads=8,casual=False,dropout=0.0):
        super(MultiHeadAttention,self).__init__()

        assert hidden_layers % no_heads == 0
        depth = hidden_layers // no_heads

        self.w_query = Dense(hidden_layers)
        self.split_reshape_query = Reshape((-1,no_heads,depth))
        self.split_permute_query = Permute((2,1,3))

        self.w_value = Dense(hidden_layers)
        self.split_reshape_value = Reshape((-1,no_heads,depth))
        self.split_permute_value = Permute((2,1,3))

        self.w_key = Dense(hidden_layers)
        self.split_reshape_key = Reshape((-1,no_heads,depth))
        self.split_permute_key = Permute((2,1,3))

        self.attention = Attention(casual=casual, dropout=dropout)
        self.join_permute_attention = Permute((2,1,3))
        self.join_reshape_attention = Reshape((-1,hidden_layers))

        self.dense = Dense(hidden_layers)
        
    def call(self,inputs,mask=None,training=None):
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v

        query = self.w_value(q)
        query = self.split_reshape_query(query)
        qyery = self.split_permute_query(query)

        value = self.w_value(v)
        value = self.split_reshape_value(value)
        value = self.split_permute_value(value)

        key = self.w_key(k)
        key = self.split_reshape_key(key)
        key = self.split_permute_key(key)

        if mask is not None:
            if mask[0] is not None:
                mask[0] = Reshape((-1,1))(mask[0])
                mask[0] = Permute((2,1))(mask[0])
            if mask[1] is not None:
                mask[1] = Reshape((-1,1))(mask[1])
                mask[1] = Permute((2,1))(mask[1])
        
        attention = self.attention([query,value,key], mask=mask)
        attention = self.join_permute_attention(attention)
        attention = self.join_reshape_attention(attention)

        x = self.dense(attention)

        return x



























