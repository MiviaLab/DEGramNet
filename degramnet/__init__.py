import numpy as np
import tensorflow as tf

from .resnet.resnet import ECAResNet152
from .layers.sincgram.sincgram import SincGram
from .layers import ZScoreNormalization
from .layers.attention.se import SqueezeAndExciteLayer

from keras.models import Model
from keras.layers import Input, Lambda, Dropout, Dense, Activation, GlobalAveragePooling2D


sample_rate = 16000.0
stft_window_seconds = 0.032
stft_hop_seconds = 0.010
n_filters: int = 128
lower_edge_hertz = 125.0
upper_edge_hertz = 7500.0


def get_degramnet(
    input_shape,
        window_order=5,
        attention=True,
        znorm_freq=False,
        get_full_model=True,
        n_classes=None,
        add_dropout=0.0,
):
    input_batch_shape = (None,)+input_shape

    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    window_length_samples = int(round(sample_rate * stft_window_seconds))
    hop_length_samples = int(round(sample_rate * stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1

    waveform = Input(shape=input_shape, dtype=tf.float32)

    squeeze_layer = Lambda(lambda x: tf.squeeze(x, axis=-1))
    waveform_squeezed = squeeze_layer(waveform)
    waveform_squeezed_shape = squeeze_layer.compute_output_shape(
        input_batch_shape)

    stft_layer = Lambda(lambda x:  tf.abs(tf.signal.stft(
        x,
        frame_length=window_length_samples,
        frame_step=hop_length_samples,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )))

    stft_op = stft_layer(waveform_squeezed)
    stft_op_squeezed_shape = stft_layer.compute_output_shape(
        waveform_squeezed_shape)

    sincgram_layer = SincGram(filters=n_filters,
                              order=window_order,
                              initializer='mel',
                              num_spectrogram_bins=num_spectrogram_bins,
                              sample_rate=sample_rate,
                              lower_edge_hertz=lower_edge_hertz,
                              upper_edge_hertz=upper_edge_hertz)

    sincgram_spec = sincgram_layer(stft_op)
    sincgram_spec_shape = sincgram_layer.compute_output_shape(
        stft_op_squeezed_shape)

    log_layer = Lambda(lambda x: tf.math.log(x+1e-07))

    features = log_layer(sincgram_spec)
    features_shape = log_layer.compute_output_shape(sincgram_spec_shape)

    axs = [-1] if znorm_freq else [-1, -2]
    spect_norm = ZScoreNormalization(
        axis=axs, scale_variance=True)
    features = spect_norm(features)

    if attention:
        attention_layer = SqueezeAndExciteLayer(
            ratio=1./16, data_format="channels_last")

        features = attention_layer(features)

    if get_full_model:

        features = Lambda(lambda x: tf.expand_dims(
            x, axis=-1), name="channel_expand")(features)
        features_shape = features_shape+(1,)

        net = ECAResNet152(input_shape=features_shape[1:],
                           input_tensor=None,
                           weights=None,  # "imagenet" if pretrained else None,
                           include_top=False)
        net.name = "backbone"

        net_out = net(features)

        pooling_layer = GlobalAveragePooling2D(name='avg_pool')
        net_out = pooling_layer(net_out)

        if add_dropout != 0.0:
            dropout_layer = Dropout(add_dropout)
            net_out = dropout_layer(net_out)

        logits = Dense(
            units=n_classes, use_bias=True)(net_out)
        predictions = Activation(
            name="preds",
            activation="softmax" if n_classes > 1 else "sigmoid")(logits)
    else:
        predictions = features

    preprocessing_model = Model(inputs=waveform, outputs=predictions)

    return preprocessing_model
