import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        qencode = tf.expand_dims(qencode, axis=1)

        c2q_att = tf.keras.activations.softmax(similarity, axis=-1)
        c2q_att = tf.expand_dims(c2q_att, axis=-1)
        c2q_att = tf.math.reduce_sum(c2q_att * qencode, -2)

        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        max_similarity = tf.math.reduce_max(similarity, axis=-1)
        c2q_att = tf.keras.activations.softmax(max_similarity)
        c2q_att = tf.expand_dims(c2q_att, axis=-1)

        weighted_sum = tf.math.reduce_sum(c2q_att * cencode, axis=-2)
        weighted_sum = tf.expand_dims(weighted_sum, 1)

        num_repeat = cencode.shape[1]

        q2c_att = tf.tile(weighted_sum, [1, num_repeat, 1])

        return q2c_att
