import tensorflow as tf

class Model(object):
    def __init__(self, config, word_mat=None, graph = None):
        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            # self.config.max_words = 20

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)

            self.seq_in = tf.placeholder(tf.int32, [None, self.config.max_len], name='seq_in')
            self.y = tf.placeholder(tf.int32, [None, 2], name='y')

            # mask = tf.cast(seq_in, tf.bool)
            # mask_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
            # mask = tf.cast(mask, tf.int32)
            TruncatedNormal = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
            l1_l2= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
            if word_mat is None:
                emb_in = tf.keras.layers.Embedding(input_dim=self.config.max_words, output_dim=self.config.emb_dim,
                                                   input_length=self.config.max_len,trainable=True)(self.seq_in)

            else:
                emb_in = tf.keras.layers.Embedding(input_dim =self.config.max_words,output_dim = self.config.emb_dim,
                                                    weights=[word_mat],input_length=self.config.max_len,
                                                    trainable = False)(self.seq_in)

            layer_cnn_0 = tf.keras.layers.Conv1D(filters=self.config.emb_dim, kernel_size=self.config.emb_dim,strides=1,
                                                 padding='same', use_bias=False, kernel_initializer=TruncatedNormal,
                                                 bias_initializer='zeros', kernel_regularizer=l1_l2)(emb_in)

            with tf.variable_scope("CNN_Layer"):
                layer_satt0 = self.self_attention(layer_cnn_0, num_heads=10, dim_head=100)

                layer_dense0 = tf.keras.layers.Conv1D(filters=self.config.emb_dim, kernel_size=10*100,
                                                     strides=1,padding='same', use_bias=True, kernel_initializer=TruncatedNormal,
                                                     bias_initializer='zeros', kernel_regularizer=l1_l2)(layer_satt0)

                layer_residual0 = layer_cnn_0+layer_dense0

                layer_out0 = self.layer_norm(layer_residual0)
                layer_out0 = tf.keras.layers.Activation('relu')(layer_out0)
                layer_pool0 = tf.keras.layers.MaxPooling1D(pool_size=3,padding='valid')(layer_out0)
                layer_cnn_0 = tf.keras.layers.Conv1D(filters=64, kernel_size=self.config.emb_dim,strides=1,
                                                     padding='same', use_bias=True, kernel_initializer=TruncatedNormal,
                                                     bias_initializer='zeros', kernel_regularizer=l1_l2)(layer_pool0)



                layer_satt1 = self.self_attention(layer_cnn_0, num_heads=20, dim_head=20)

                layer_Dense1 = tf.keras.layers.Conv1D(filters=64, kernel_size=20*20,strides=1,
                                                     padding='same', use_bias=False, kernel_initializer=TruncatedNormal,
                                                     bias_initializer='zeros', kernel_regularizer=l1_l2)(layer_satt1)

                layer_residual1 = layer_cnn_0+layer_Dense1

                layer_out1 = self.layer_norm(layer_residual1)
                layer_out1 = tf.keras.layers.Activation('relu')(layer_out1)
                layer_pool1 = tf.keras.layers.MaxPooling1D(pool_size=3,padding='valid')(layer_out1)

                layer_flatten = tf.keras.layers.Flatten()(layer_pool1)

                self.logits = tf.keras.layers.Dense(2,name='logits')(layer_flatten)



            with tf.variable_scope("Output_Layer"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels =self.y)
                self.loss = tf.reduce_mean(cross_entropy)

                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.logits, 1))

                self.acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                tf.add_to_collection('loss',  self.loss)
                tf.add_to_collection('acc', self.acc)


                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('acc', self.acc)
                self.merged = tf.summary.merge_all()

                self.train_step = tf.train.AdamOptimizer().minimize(self.loss)




    def self_attention(self,input_tensor,num_heads,dim_head):

        TruncatedNormal = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
        l1_l2= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)


        input_tensor = self.layer_norm(input_tensor)
        input_tensor = tf.keras.layers.Activation('relu')(input_tensor)

        layer_cnn_0 = tf.keras.layers.Conv1D(filters = num_heads*dim_head, kernel_size=self.config.emb_dim, strides=1,
                                             padding='same',use_bias=True, kernel_initializer=TruncatedNormal,
                                             bias_initializer='zeros',kernel_regularizer=l1_l2)(input_tensor)
        Q = layer_cnn_0
        K = layer_cnn_0
        V = layer_cnn_0

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        seq_len = K_.get_shape().as_list()[-2]

        att_matrix = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        att_matrix = att_matrix / (dim_head ** 0.5)

        # Key Masking
        mask = tf.sign(tf.abs(tf.reduce_sum(input_tensor, axis=-1)))  # (N, T_k)

        mask_matrix = tf.tile(mask, [num_heads, 1])
        mask_matrix = tf.tile(tf.expand_dims(mask_matrix, axis=1), [1, seq_len, 1])
        paddings = tf.ones_like(mask_matrix) * (-2 ** 31 + 1)
        att_matrix = tf.where(tf.equal(mask_matrix, 0), tf.cast(paddings,tf.float32), att_matrix)

        attention = tf.nn.softmax(att_matrix)

        outputs = tf.matmul(attention,V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 )

        outputs = self.layer_norm(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)

        return outputs

    def layer_norm(self,input_tensor, name=None):
        """Run layer normalization on the last dimension of the tensor."""
        return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)