import tensorflow as tf

class Model(object):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,config,word_mat=None,graph=None):
        # self, sequence_length, , vocab_size,
        # embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0

        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.config = config
            global_step = tf.Variable(0, name="global_step", trainable=False)
            TruncatedNormal = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
            l1_l2 = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
            sequence_length = config.max_len
            filter_sizes=[3,4,5]
            embedding_size = 300
            num_filters = 128

            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.y = tf.placeholder(tf.float32, [None, 2], name="y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


            # # Embedding layer
            emb_in = tf.keras.layers.Embedding(input_dim=self.config.max_words, output_dim=self.config.emb_dim,
                                               weights=[word_mat], input_length=self.config.max_len,
                                               trainable=False)(self.input_x)

            encoder = tf.expand_dims(emb_in, -1)
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    layer_conv = tf.keras.layers.Conv2D(filters=num_filters,kernel_size=[filter_size,embedding_size],
                                                        padding='valid',name="conv",
                                                        kernel_initializer =TruncatedNormal,
                                                        kernel_regularizer=l1_l2 )(encoder)


                    layer_conv = tf.keras.layers.BatchNormalization()(layer_conv)
                    layer_conv = self.gelu(layer_conv)

                    layer_pooling = tf.keras.layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1), strides=[1,1],
                                                                 padding='valid',data_format=None)(layer_conv)

                    pooled_outputs.append(layer_pooling)


            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            layer_conv_out = tf.concat(pooled_outputs, axis = 3)
            layer_conv_out = tf.reshape(layer_conv_out, [-1, num_filters_total])


            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):

                self.logits= tf.keras.layers.Dense(2,kernel_initializer=TruncatedNormal,
                                                   kernel_regularizer=l1_l2)(layer_conv_out)

                self.softmax_output=tf.keras.layers.Activation('softmax')(self.logits)

                self.predictions = tf.argmax(self.logits, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss = tf.reduce_mean(losses)

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="acc")

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                tf.add_to_collection('loss', self.loss)
                tf.add_to_collection('acc', self.acc)

                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('acc', self.acc)
                tf.summary.histogram('softmax_output', self.softmax_output)
                self.merged = tf.summary.merge_all()

    def gelu(self,input_tensor):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415

        Args:
          input_tensor: float Tensor to perform activation.

        Returns:
          `input_tensor` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf
