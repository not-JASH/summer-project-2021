from tensorflow.keras.layers import Input, Dense, Dropout, Add, Layer,Embedding,  Reshape, Permute, Attention, LayerNormalization
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy, RootMeanSquaredError  
from tensorflow.math import sqrt, rsqrt, minimum, logical_not, equal
from tensorflow import cast, float32,shape, reduce_sum, convert_to_tensor
from numpy import power, float32 as f32, arange, sin, cos, newaxis,array
from math import floor

'''
    This transformer pulled and adapted from
    https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83
    https://colab.research.google.com/drive/1CBe2VlogbyXzmIyRQGH5xzuvLwGrvjcf?usp=sharing#scrollTo=6wOZpgGc8hSH
    Credit: Max Gerber

    d_model -> d_model
'''

class Transformer:
    def __init__(self,input_size,target_size,no_layers=4,d_model=128,dff=512,no_heads=8,dropout_rate=0.1):
        input = Input(shape=(None,))
        target = Input(shape=(None,))

        encoder = Encoder(input_size, no_layers=no_layers,d_model=d_model,no_heads=no_heads,dff=dff,dropout=dropout_rate)
        decoder = Decoder(target_size,no_layers=no_layers,d_model=d_model,no_heads=no_heads,dff=dff,dropout=dropout_rate)

        x = encoder(input)
        x = decoder([target,x], mask=encoder.compute_mask(input))
        x = Dense(target_size)(x)

        self.model = Model(inputs=[input,target], outputs=x)

        self.model.summary()

        self.d_model = d_model
        #return model

    def train(self,train_dataset,val_dataset,batch_size=10):
        optimizer = Adam(CustomSchedule(self.d_model),beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        #loss = SparseCategoricalCrossentropy(from_logits=True,reduction='none')
        loss = MeanSquaredError(reduction='none')

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
            no_batches = batch

        no_val_batches = 0
        for (batch,(_,_)) in enumerate(val_dataset):
            no_val_batches = batch

        def generator(dataset):
            while True:
                for x_batch,y_batch in dataset:
                    #x_batch,y_batch = convert_to_tensor(x_batch,float32),convert_to_tensor(y_batch,float32)
                    yield ([x_batch, y_batch[:,:-1]], y_batch[:,1:])

        history = self.model.fit(x = generator(train_dataset),
                                 validation_data=generator(val_dataset),
                                 steps_per_epoch=no_batches,
                                 validation_steps=no_val_batches,
                                 epochs=100,    
                                 verbose=1
                                 )
        return history


class CustomSchedule(LearningRateSchedule):
    def __init__(self,d_model,warmup_steps=4000):
        super(CustomSchedule,self).__init__()

        self.d_model = d_model
        self.d_model = cast(self.d_model,float32)

        self.warmup_steps = warmup_steps

    def __call__(self,step):
        return rsqrt(self.d_model) * minimum(rsqrt(step),step*(self.warmup_steps ** -1.5))


class Encoder(Layer):
    def __init__(self,input_size,no_layers=4,d_model=512,no_heads=8,dff=2048,maximum_position_encoding=10000,dropout=0.0):
        super(Encoder,self).__init__()

        self.d_model = d_model
        self.embedding = Embedding(input_size,d_model,mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding, d_model)
        self.encoder_layers = [ EncoderLayer(d_model=d_model,no_heads=no_heads,dff=dff,dropout=dropout) for _ in range(no_layers)]
        self.dropout = Dropout(dropout)

    def call(self,inputs,mask=None,training=None):
        x = self.embedding(inputs)
        
        #positional encoding
        x *= sqrt(cast(self.d_model,float32))
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
    def __init__(self,target_size,no_layers=4,d_model=512,no_heads=8,dff=2048,maximum_position_encoding=10000,dropout=0.0):
        super(Decoder,self).__init__()

        self.d_model = d_model
        self.embedding = Embedding(target_size,d_model,mask_zero=True)
        self.pos = positional_encoding(maximum_position_encoding,d_model)

        self.decoder_layers = [ DecoderLayer(d_model=d_model, no_heads=no_heads, dff=dff, dropout=dropout) for _ in range(no_layers)]

        self.dropout = Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        x = self.embedding(inputs[0])
        #positional encoding
        x *= sqrt(cast(self.d_model,float32))
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
    def __init__(self,d_model=512,no_heads=8,dff=2048,dropout=0.0):
        super(EncoderLayer,self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model,no_heads)
        self.dropout_attention = Dropout(dropout)
        self.add_attention = Add()
        self.layer_norm_attention = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
        self.dropout_dense = Dropout(dropout)
        self.add_dense = Add()
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
        x = self.layer_norm_dense(x)

        return x

class DecoderLayer(Layer):
    def __init__(self,d_model=512,no_heads=8,dff=2048,dropout=0.0):
        super(DecoderLayer,self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model,no_heads,causal=True)
        self.dropout_attention1 = Dropout(dropout)
        self.add_attention1 = Add()
        self.layer_norm_attention1 = LayerNormalization(epsilon=1e-6)

        self.multi_head_attention2 = MultiHeadAttention(d_model,no_heads)
        self.dropout_attention2 = Dropout(dropout)
        self.add_attention2 = Add()
        self.layer_norm_attention2 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(dff,activation='relu')
        self.dense2 = Dense(d_model)
        self.dropout_dense = Dropout(dropout)
        self.add_dense = Add()
        self.layer_norm_dense = LayerNormalization(epsilon=1e-6)

    def call(self,inputs,mask=None,training=None):
        # print(mask)
        attention = self.multi_head_attention1([inputs[0],inputs[0],inputs[0]], mask = [mask[0],mask[0]])
        attention = self.dropout_attention1(attention,training=training)
        x = self.add_attention1([inputs[0],attention])
        x = self.layer_norm_attention1(x)

        attention = self.multi_head_attention2([x,inputs[1],inputs[1]], mask=[mask[0],mask[0]])
        attention = self.dropout_attention2(attention,training=training)
        x = self.add_attention1([x,attention]) #self.add_attention2 ?
        x = self.layer_norm_attention1(x) #self.layer_norm_attention1 ?

        ## Feed  Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense,training=training)
        x = self.add_dense([x,dense])
        x = self.layer_norm_dense(x)
        
        return x

class MultiHeadAttention(Layer):
    def __init__(self,d_model=512,no_heads=8,causal=False,dropout=0.0):
        super(MultiHeadAttention,self).__init__()

        assert d_model % no_heads == 0
        depth = d_model // no_heads

        self.w_query = Dense(d_model)
        self.split_reshape_query = Reshape((-1,no_heads,depth))
        self.split_permute_query = Permute((2,1,3))

        self.w_value = Dense(d_model)
        self.split_reshape_value = Reshape((-1,no_heads,depth))
        self.split_permute_value = Permute((2,1,3))

        self.w_key = Dense(d_model)
        self.split_reshape_key = Reshape((-1,no_heads,depth))
        self.split_permute_key = Permute((2,1,3))

        self.attention = Attention(causal=causal, dropout=dropout)
        self.join_permute_attention = Permute((2,1,3))
        self.join_reshape_attention = Reshape((-1,d_model))

        self.dense = Dense(d_model)
        
    def call(self,inputs,mask=None,training=None):
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v

        query = self.w_value(q)
        query = self.split_reshape_query(query)
        query = self.split_permute_query(query)

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

def get_angles(pos, i, d_model):
    angle_rates = 1/power(10000,(2*(i/2))/f32(d_model))
    return pos*angle_rates

def positional_encoding(position,d_model):
    angle_rads = get_angles(arange(position)[:,newaxis], arange(d_model)[newaxis,:], d_model)

    #apply sin to even indicies in array; 2i
    angle_rads[:,0::2] = sin(angle_rads[:,0::2])

    #apply cos to odd indicies in array; 2i+1
    angle_rads[:,1::2] = cos(angle_rads[:,1::2])
    pos_encoding = angle_rads[newaxis,...]
    
    return cast(pos_encoding,dtype=float32)























