from keras.layers import Layer, Embedding, Dropout, Add, LayerNormalization, Dense, Reshape, Permute, Attention
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from keras.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from keras.models import Model
from keras.metrics import SparseCategoricalAccuracy, RootMeanSquaredError
from keras.engine.input_layer import InputLayer as Input
from tensorflow.math import sqrt, rsqrt, minimum, logical_not, equal
from tensorflow import cast, float32,shape, reduce_sum

'''
    This transformer pulled and adapted from
    https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83
    https://colab.research.google.com/drive/1CBe2VlogbyXzmIyRQGH5xzuvLwGrvjcf?usp=sharing#scrollTo=6wOZpgGc8hSH
    Credit: Max Gerber

    d_model -> hidden_layers
'''

class Transformer:
    def __init__(self,input_size,target_size,no_layers=4,hidden_layers=128,dff=512,no_heads=8,dropout_rate=0.1):
        input = Input(shape=(None,))
        target = Input(shape=(None,))

        encoder = Encoder(input_size, no_layers=no_layers,hidden_layers=hidden_layers,no_heads=no_heads,dff=dff,dropout=dropout_rate)
        decoder = Decoder(target_size,no_layers=no_layers,hidden_layers=hidden_layers,no_heads=no_heads,dff=dff,dropout_rate=dropout_rate)

        x = encoder(input)
        x = decoder([target,x], mask=encoder.compute_mask(input))
        x = Dense(target_size)(x)

        self.model = Model(inputs=[input,target], outputs=x)

        self.model.summary()

        self.hidden_layers = hidden_layers
        #return model

    def train(self,train_dataset,val_dataset):
        optimizer = Adam(CustomSchedule(self.hidden_layers),beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        #loss = SparseCategoricalCrossentropy(from_logits=True,reduction='none')
        loss = MeanSquaredError()

        def masked_loss(y_true,y_pred):
            mask = logical_not(equal(y_true,0))
            _loss = loss(y_true,y_pred)

            mask = cast(mask, dtype=_loss.dtype)
            _loss *= mask

            return reduce_sum(_loss)/reduce_sum(mask)

        #metrics = [loss,masked_loss,SparseCategoricalAccuracy()]
        metrics = [loss,masked_loss,RootMeanSquaredError()]

        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics) #masked

        no_batches = 0
        for (batch,(_,_)) in enumerate(train_dataset):
            val_batches = batch

        def generator(data_set):
            while True:
                for x_batch, y_batch in dataset:
                    yield([x_batch, en_batch[:,:-1]], en_batch[:,1:])

        history = self.model.fit(x = generator(train_dataset),
                                 validation_data=generator(val_dataset),
                                 epochs=20,
                                 steps_per_epoch=num_batches,
                                 validation_steps=val_batches)
        return history


class CustomSchedule(LearningRateSchedule):
    def __init__(self,hidden_layers,warmup_steps=4000):
        super(CustomSchedule,self).__init__()

        self.hidden_layers = hidden_layers
        self.hidden_layers = cast(self.hidden_layers,float32)

        self.warmup_steps = warmup_steps

    def __call__(self,step):
        return rsqrt(self.hidden_layers) * minimum(rsqrt(step),step*(self.warmup_steps ** -1.5))


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



























