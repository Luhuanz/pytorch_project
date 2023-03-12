import tensorflow as tf

class Highway(tf.keras.layers.Layer):

    def __init__(self, activation='relu', *args, **kwargs):
        self.activation = activation
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        bias_init = tf.keras.initializers.Constant(-1)
        self.dense1 = tf.keras.layers.Dense(dim, bias_initializer=bias_init)
        self.dense2 = tf.keras.layers.Dense(dim)

        self.dense1.build(input_shape)
        self.dense2.build(input_shape)

        super().build(input_shape)

    def call(self, x):
        dim = x.shape[-1]

        transform_gate = self.dense1(x)
        transform_gate = tf.keras.layers.Activation('sigmoid')(transform_gate)

        carry_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense2(x)
        transformed_data = tf.keras.layers.Activation(self.activation)(transformed_data)
        transformed_gated = tf.keras.layers.multiply([transform_gate, transformed_data])
        identity_gated = tf.keras.layers.multiply([carry_gate, x])
        value = tf.keras.layers.add([transformed_gated, identity_gated])

        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config
