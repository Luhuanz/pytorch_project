import tensorflow as tf

class MergedContext(tf.keras.layers.Layer):

    def call(self, cencode, c2q_att, q2c_att):
        m1 = cencode * c2q_att
        m2 = cencode * q2c_att

        concat = tf.keras.layers.concatenate([
            cencode, c2q_att, m1, m2],
            axis=-1
        )

        return concat

    def compute_output_shape(self, input_shape):
        shape, _, _ = input_shape
        return shape[:-1] + (shape[-1] * 4, )
