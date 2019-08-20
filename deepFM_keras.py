from keras.layers import Dense, Concatenate, Lambda, Add, Dot, Activation
from keras.engine.topology import Layer
from keras import backend as K

__author__ = "github/mfulford"

class LayerDeepFM(Layer):
    """
    params:

    deep : bool, default = True.
        Whether or not to include deep neural net component
        (For deepFM, deep and fm must both be set to True)

    fm : default = True.
        Whehter or not to include FM component.
        (For deepFM, deep and fm must both be set to True)

    return_embedding: bool, default = False
        True:
            returns the embedding vector (deep, fm or deepfm)
                for deep =  True and fm = True,  returns concatenated deep + fm vectors
                for deep =  True and fm = False, returns deep vector
                for deep = False and fm = True,  returns fm vector
        False:
            returns output (which needs to be passed into the output neuron)

    field_lengths: dictionary of input field dimensions.
        e.g with 5 fields with dimensions 3,3,2,2,2 respectively:
            field_lengths = {0:3, 1:3, 2:2, 3:2, 4:2}

    one_hot: bool, default = True
        True:
            input is one hot encoded
            e.g. [1,0,0, 0,0,1, 1,0, 1,0, 1,0]
            input has length = sum(field_lengths.values())
        False:
            input is integer encoded
            e.g [1, 3, 1, 1, 1]
            input has length = len(field_lengths)

            NB  integer input value is from 1 to num options.
                value of 0 is equivalent to no options selected.

    embedding_size: int, length of embedding vector

    deepHidenLayer: Default = [50, 50]. Number of neurons in deep layers 1 and 2.

    """

    def __init__(self, field_lengths, embedding_size, one_hot=True, deepHidenLayers=[50, 50], deep=True, fm=True,
                 return_embedding=False, l2_reg=0., dropout=0., **kwargs):

        assert isinstance( deep, bool)
        assert isinstance( fm, bool)

        self.deep = deep
        self.fm = fm

        self.return_embedding = return_embedding
        self.one_hot = one_hot
        self.field_lengths = field_lengths
        self.num_fields = len(field_lengths)
        self.embedding_size = embedding_size

        # deep hidden layer number of neurons (assumes 2 hidden layers)
        self.h1 = deepHidenLayers[0]
        self.h2 = deepHidenLayers[1]

        # function to split input into fields
        # the order of field_lengths dict must correspond to input format
        def start_stop_indices(field_lengths, num_fields):
            len_field = {}
            id_start = {}
            id_stop = {}
            len_input = 0

            for i in range(1, num_fields + 1):
                len_field[i] = field_lengths[i - 1]
                id_start[i] = len_input
                len_input += len_field[i]
                id_stop[i] = len_input

            return len_field, len_input, id_start, id_stop

        self.len_field, self.len_input, self.id_start, self.id_stop = \
            start_stop_indices(self.field_lengths,self.num_fields)


        # get all field dot product combinations
        # can be done with itertools.combinations ( range(1,num_fields), 2)
        def Field_Combos(num_fields):
            field_list = list(range(1, num_fields))  # 1 to num_fields - 1
            combo = {}
            combo_count = 0
            for idx, field in enumerate(field_list):
                sub_list = list(range(field + 1, num_fields + 1))  # field+1 to num_fields
                combo_count += len(sub_list)
                combo[field] = sub_list

            return combo, combo_count

        self.field_combinations, self.combo_count = Field_Combos(self.num_fields)

        # determine output dimensions
        if self.return_embedding:
            if self.deep and self.fm:
                # deep Out, 2nd order FM, 1st order FM (sum)
                self.output_dim = self.h2 + self.combo_count + 1
            elif self.fm:
                self.output_dim = self.combo_count + 1
            elif self.deep:
                self.output_dim = self.h2
        else:
            self.output_dim = 1

        super(LayerDeepFM, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.embed_W = {}
        self.embed_b = {}
        total_embed_size = 0

        for i in range(1, self.num_fields + 1):
            input_dim = self.len_field[i]
            _name = "embed_W" + str(i)
            self.embed_W[i] = self.add_weight(shape=(input_dim, self.embedding_size), initializer='glorot_uniform', name=_name)
            _name = "embed_b" + str(i)
            self.embed_b[i] = self.add_weight(shape=(self.embedding_size,), initializer='zeros', name=_name)
            total_embed_size += self.embedding_size

        self.h1_W = self.add_weight(shape=(total_embed_size, self.h1), initializer='glorot_uniform', name='h1_W')
        self.h1_b = self.add_weight(shape=(self.h1,), initializer='zeros', name='h1_b')

        self.h2_W = self.add_weight(shape=(self.h1, self.h2), initializer='glorot_uniform', name='h2_W')
        self.h2_b = self.add_weight(shape=(self.h2,), initializer='zeros', name='h2_b')

        self.FM_add_W = self.add_weight(shape=(input_shape[1], 1), initializer='glorot_uniform', name='FM_add_W')
        self.WX_W = self.add_weight(shape=(self.h2, 1), initializer='glorot_uniform', name='WX_W')
        self.WX_b = self.add_weight(shape=(1,), initializer='zeros', name='WX_b')

        super(LayerDeepFM, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, inputTensor):
        latent = {}
        if self.one_hot:
            # sparse field embedding
            sparse = {}
            for i in range(1, self.num_fields + 1):
                sparse[i] = Lambda(lambda x: x[:, self.id_start[i]:self.id_stop[i]],
                                   output_shape=((self.len_field[i],)))(inputTensor)
                latent[i] = Activation('linear')(K.bias_add(K.dot(sparse[i],self.embed_W[i]),self.embed_b[i]))
        else:
            for i in range(1, self.num_fields + 1):
                inp = K.cast(inputTensor[:, i - 1], dtype=K.tf.int32)
                inp_to_1hot_wzero = K.one_hot(inp,self.len_field[i] + 1) # len_field + 1 as one_hot encode encodes val = 0
                inp_to_1hot = inp_to_1hot_wzero[:, 1:]  # remove 1st col (for 0)
                # e.g. len_field = 3

                #       inp = 0 --> inp_to_1hot_wzero = [1, 0, 0, 0]

                #                   inp_to_1hot = [0, 0, 0]

                #       inp = 3 --> inp_to_1hot_wzero = [0, 0, 0, 1]

                #                   inp_to_1hot = [0, 0, 1]
                latent[i] = Activation('linear')(K.bias_add(K.dot(inp_to_1hot, self.embed_W[i]), self.embed_b[i]))

        ConcatLatent = Concatenate()(list(latent.values()))

        # deep
        Deep_h1 = Activation('relu')(K.bias_add(K.dot(ConcatLatent,self.h1_W),self.h1_b)) # hidden layer 1 of deep component
        Deep_h2 = Activation('relu')(K.bias_add(K.dot(Deep_h1, self.h2_W), self.h2_b)) # hidden layer 2 of deep component

        # FM part (1st order)
        FM_add = Activation('linear')(K.dot(inputTensor, self.FM_add_W))

        # FM 2nd order
        dot_latent = {}
        dot_cat = []
        for i in range(1, self.num_fields):
            dot_latent[i] = {}
            for f in self.field_combinations[i]:
                dot_latent[i][f] = Dot(axes=-1, normalize=False)([latent[i], latent[f]])
                dot_cat.append(dot_latent[i][f])

        ConcatDot = Concatenate()(dot_cat)
        ConcatFM = Concatenate()([FM_add, ConcatDot])  # length= 1 + num_combinations


        # if want to return the deep or fm or deepfm embedding vectors
        # NB returning deepfm embedding won't be the same as deepFM
        # uses learned weight with linear activation for deep embed and
        # weight =1 (untrained for fm)
        if self.return_embedding:
            if self.deep and self.fm:
                EmbedOut = Concatenate()([FM_add, ConcatDot, Deep_h2])
            elif self.fm:
                EmbedOut = Concatenate()([FM_add, ConcatDot])
            elif self.deep:
                EmbedOut = Deep_h2
            return EmbedOut

        # Deep:  W * x (x = neuron value deep_h2, w = trainable weight)
        WX = Activation('linear')(K.bias_add(K.dot(Deep_h2, self.WX_W),self.WX_b))

        # FM: (weight =1 - non-trainable weight)
        SumFM = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(ConcatFM)
        SumFM_WX = Add()([WX, SumFM])  # Sum deep + fm  (no weights)

        if self.deep and self.fm:
            out = SumFM_WX
        elif self.fm:
            out = SumFM
        elif self.deep:
            out = WX

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)