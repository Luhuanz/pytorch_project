import tensorflow as tf

class SpanBegin(tf.keras.layers.Layer):

    def build(self, input_shape):
        last_dim = input_shape[0][-1] + input_shape[1][-1]
        inn_shape_dense1 = input_shape[0][:-1] + (last_dim, )
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense1.build(inn_shape_dense1)

        super().build(input_shape)

    def call(self, inputs):
        merged_ctx, modeled_ctx = inputs

        span_begin_inn = tf.concat([merged_ctx, modeled_ctx], axis=-1)
        span_begin_weight = tf.keras.layers.TimeDistributed(self.dense1)(span_begin_inn)
        span_begin_weight = tf.squeeze(span_begin_weight, axis=-1)
        span_begin_prob = tf.keras.activations.softmax(span_begin_weight)

        return span_begin_prob

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1]

class SpanEnd(tf.keras.layers.Layer):

    def build(self, input_shape):
        emb_size = input_shape[0][-1] // 2
        inn_shape_bilstm = input_shape[0][:-1] + (emb_size * 14, )
        inn_shape_dense = input_shape[0][:-1] + (emb_size * 10, )

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(emb_size, return_sequences=True))
        self.bilstm.build(inn_shape_bilstm)

        self.dense = tf.keras.layers.Dense(1)
        self.dense.build(inn_shape_dense)

        super().build(input_shape)

    def call(self, inputs):
        cencode, merged_ctx, modeled_ctx, span_begin_prob = inputs

        _span_begin_prob = tf.expand_dims(span_begin_prob, axis=-1)
        weighted_sum = tf.math.reduce_sum(_span_begin_prob * modeled_ctx, axis=-2)

        weighted_ctx = tf.expand_dims(weighted_sum, axis=1)
        tile_shape = tf.concat([[1], [cencode.shape[1]], [1]], axis=0)
        weighted_ctx = tf.tile(weighted_ctx, tile_shape)
        m1 = modeled_ctx * weighted_ctx

        span_end_repr = tf.concat([merged_ctx, modeled_ctx, weighted_ctx, m1], axis=-1)
        span_end_repr = self.bilstm(span_end_repr)
        span_end_inn = tf.concat([merged_ctx, span_end_repr], axis=-1)
        span_end_weights = tf.keras.layers.TimeDistributed(self.dense)(span_end_inn)
        span_end_prob = tf.keras.activations.softmax(tf.squeeze(span_end_weights, axis=-1))

        return span_end_prob

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1]

class Combine(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.stack(inputs, axis=1)
