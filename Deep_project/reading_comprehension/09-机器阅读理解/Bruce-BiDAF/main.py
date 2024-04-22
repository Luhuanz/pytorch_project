import os
import logging
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import layers
import preprocess
import numpy as np

class BiDAF:
    def __init__(
        self, clen, qlen, emb_size,
        max_features=5000,
        num_highway_layers=2,
        encoder_dropout=0,
        num_decoders=2,
        decoder_dropout=0,
    ):
        self.clen = clen
        self.qlen = qlen
        self.max_features = max_features
        self.emb_size = emb_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    def build_model(self):
        cinn = tf.keras.layers.Input(shape=(self.clen,), name='CInn')
        qinn = tf.keras.layers.Input(shape=(self.qlen,), name='QInn')

        embedding_layer = tf.keras.layers.Embedding(self.max_features, self.emb_size)
        cemb = embedding_layer(cinn)
        qemb = embedding_layer(qinn)

        for i in range(self.num_highway_layers):
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')

            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)
        qencode = encoder_layer(qemb)

        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [cinn, qinn]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )

def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)

def accuracy(y_true, y_pred):

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)

if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './data/train.json',
        './data/dev.json',
        './data/test.json'
    ])

    train_c, train_q, train_y = ds.get_dataset('./data/train.json')
    test_c, test_q, test_y = ds.get_dataset('./data/dev.json')

    print(train_c.shape, train_q.shape, train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=128,
        max_features=len(ds.charset)
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_c, train_q], train_y,
        batch_size=16,
        epochs=100,
        validation_data=([test_c, test_q], test_y)
    )
