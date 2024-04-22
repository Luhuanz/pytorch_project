import tensorflow as tf

class Similarity(tf.keras.layers.Layer):

    def build(self, input_shape):
        word_vector_dim = input_shape[0][-1]
        weight_vector_dim = word_vector_dim * 3

        self.kernel = self.add_weight(
            name='SimilarityWeight',
            shape=(weight_vector_dim, 1),
            initializer='uniform',
            trainable=True,
        )

        self.bias = self.add_weight(
            name='SimilarityBias',
            shape=(),
            initializer='ones',
            trainable=True,
        )

        super().build(input_shape)

    def compute_similarity(self, repeated_cvectors, repeated_qvectors):
        element_wise_multiply = repeated_cvectors * repeated_qvectors

        concat = tf.keras.layers.concatenate([
            repeated_cvectors,
            repeated_qvectors,
            element_wise_multiply
        ], axis=-1)

        dot_product = tf.tensordot(concat, self.kernel, axes=1)
        dot_product = tf.squeeze(dot_product, axis=-1)

        return tf.keras.activations.linear(dot_product + self.bias)

    def call(self, inputs):
        c_vector, q_vector = inputs

        n_cwords = c_vector.shape[1]
        n_qwords = q_vector.shape[1]

        cdim_repeat = tf.convert_to_tensor([1, 1, n_qwords, 1])
        qdim_repeat = tf.convert_to_tensor([1, n_cwords, 1, 1])

        repeated_cvectors = tf.tile(tf.expand_dims(c_vector, axis=2), cdim_repeat)
        repeated_qvectors = tf.tile(tf.expand_dims(q_vector, axis=1), qdim_repeat)

        similarity = self.compute_similarity(repeated_cvectors, repeated_qvectors)

        return similarity
