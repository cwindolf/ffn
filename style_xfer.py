import numpy as np
import scipy.linalg as la
import tensorflow as tf
import preprocessing.data_util as dx
import models
from absl import flags
from absl import app

# ------------------------------- flags -------------------------------

# Script args
flags.DEFINE_enum(
    'method', 'wc', ['wc', 'hm'], 'Whitening-coloring or histogram matching'
)
flags.DEFINE_string('contentspec', None, 'Input datspec, the content source.')
flags.DEFINE_string('stylespec', None, 'Target style datspec.')
flags.DEFINE_string('outspec', None, 'Write output here.')

# Model flags
flags.DEFINE_integer('layer', None, 'Depth of encoder/decoder')

# Checkpoint flags
flags.DEFINE_string(
    'ffn_ckpt', None, 'If supplied, load FFN weights for the encoder.'
)
flags.DEFINE_string(
    'encoder_ckpt', None, 'If supplied, load encoder from this checkpoint.'
)
flags.DEFINE_string(
    'decoder_ckpt', None, 'Load decoder from this checkpoint.'
)


# ------------------------------- lib ---------------------------------


# TensorFlow business -------------------------------------------------


def load_encoder(
    session, layer, fov_size, seed, ffn_ckpt=None, encoder_ckpt=None
):
    # Load FFN weights ------------------------------------------------
    # Hooking graphs together... This bit loads up weights.
    if ffn_ckpt:
        assert not encoder_ckpt
        encoder = models.ConvStack3DEncoder.from_ffn_ckpt(
            ffn_ckpt=ffn_ckpt, fov_size=fov_size, input_seed=seed, depth=layer
        )

    # Model graph -----------------------------------------------------
    encoder_graph = tf.Graph()
    with encoder_graph.as_default():
        # Init encoder ------------------------------------------------
        if not ffn_ckpt:
            assert encoder_ckpt
            encoder = models.ConvStack3DEncoder(
                input_seed=seed, fov_size=fov_size, depth=layer
            )
        encoder.define_tf_graph()

        # Make init op
        if ffn_ckpt:
            einit_op = tf.initializers.variables(encoder.vars)
            session.run(einit_op)
        else:
            encoder.saver.restore(session, encoder_ckpt)

    return encoder, encoder_graph


def load_decoder(session, layer, fov_size, decoder_ckpt):
    decoder_graph = tf.Graph()
    with decoder_graph.as_default():
        # Init decoder
        decoder = models.ConvStack3Ddecoder(fov_size=fov_size, depth=layer)
        decoder.define_tf_graph()

        # Restore
        decoder.saver.restore(session, decoder_ckpt)

    return decoder, decoder_graph


# Feature transforms --------------------------------------------------


def whitening_coloring_transform(content_features, style_features):
    '''Implements the whitening-coloring transform from arXiv:1705.08086

    Whitens the feature correlations of `content_features`, and then
    colors them to match those of `style_features`.

    Arguments
    ---------
    content_features, style_features : rank-4 np.arrays
        The embedded features of the content and style volumes.

    Returns
    -------
    wct_features : 4d np.array
        The whitened and re-colored feature map.
    '''
    # Reshape everyone to features x flatspace
    orig_content_shape = content_features.shape
    content_features = content_features.transpose(3, 0, 1, 2)
    content_features = content_features.reshape(orig_content_shape[0], -1)
    style_features = style_features.transpose(3, 0, 1, 2)
    style_features = style_features.reshape(style_features.shape[0], -1)

    # Whitening transform ---------------------------------------------
    # Center
    content_feature_means = content_features.mean(axis=1, keepdims=True)
    content_features -= content_feature_means

    # Compute whitening matrix
    content_feature_corrs = content_features @ content_features.T
    cw, cv = la.eigh(content_feature_corrs)
    whitener = cv @ np.reciprocal(np.sqrt(cw)) @ cv.T

    # Apply whitening
    whitened_features = whitener @ content_features

    # Coloring transform ----------------------------------------------
    # Center
    style_feature_means = style_features.mean(axis=1, keepdims=True)
    style_features -= style_feature_means

    # Compute coloring matrix
    style_feature_corrs = style_features @ style_features.T
    sw, sv = la.eigh(style_feature_corrs)
    colorizer = sv @ np.sqrt(sw) @ sv.T

    # Apply coloring and shape back to space x features
    wct_features = colorizer @ whitened_features + style_feature_means
    wct_features = wct_features.transpose(1, 0).reshape(*orig_content_shape)

    return wct_features


def histogram_matching_transform(content_features, style_features):
    '''Implements generic channel-wise histogram matching

    (Well not so generic -- it works on 4d arrays and assumes channels
    are on the last axis.)

    Arguments
    ---------
    content_features, style_features : rank-4 np.arrays
        The embedded features of the content and style volumes.

    Returns
    -------
    hmt_features : 4d np.array
    '''
    # Reshape everyone to features x flatspace
    orig_content_shape = content_features.shape
    content_features = content_features.transpose(3, 0, 1, 2)
    content_features = content_features.reshape(orig_content_shape[0], -1)
    style_features = style_features.transpose(3, 0, 1, 2)
    style_features = style_features.reshape(style_features.shape[0], -1)

    # Match histograms channel-by-channel
    hmt_features = np.stack(
        [
            dx.match_histogram(content_channel, style_channel)
            for content_channel, style_channel in zip(
                content_features, style_features
            )
        ]
    )

    # Reshape back to space x features
    hmt_features = hmt_features.transpose(1, 0).reshape(*orig_content_shape)

    return hmt_features


# --------------------- style transfer procedure ----------------------


def feature_transform_style_xfer(
    layer,
    contentspec,
    stylespec,
    outspec,
    decoder_ckpt,
    encoder_ckpt=None,
    ffn_ckpt=None,
    method='wc',
):
    '''Style transfer for domain adapatation

    Based on Li et al, Universal Style Transfer via Feature Transforms,
    [arXiv:1705.08086].

    Arguments
    ---------
    layer : int
        This gives the depth of the encoder and decoder, which as
        of right now are always the same. If you are using the
        beginning of an FFN instead of using a separately trained
        encoder, then this is the depth at which the ffn is chopped.
    contentspec : string
        Datspec pointing to the data from the new domain.
    stylespec : string
        Datspec pointing to the data from the original domain.
    outspec : string
        Datspec in which to save the results.
    decoder_ckpt : string
        Path to checkpoint of trained decoder. It should be one that
        works with the encoder you pass in!
    encoder_ckpt : string, optional
        A tensorflow checkpoint for loading up a Convstack3DEncoder.
        Supply either this or ffn_ckpt.
    ffn_ckpt : string, optional
        A tensorflow checkpoint to load up an FFN model.
        Supply either this or encoder_ckpt.
    method : string
        Give 'wc' for whitening-coloring transform, or 'hm' for
        histogram-matching baseline transform.
    '''
    # Set up ----------------------------------------------------------
    seed = None

    # Load data -------------------------------------------------------
    content_volume = dx.loadspec(contentspec)
    style_volume = dx.loadspec(stylespec)
    if style_volume.shape != content_volume.shape:
        raise ValueError(
            'Style and content volumes were different shapes. This case '
            'can be supported but it will take a little work. Maybe not '
            'too much work actually.'
        )
    fov_size = content_volume.shape

    # Load encoder ----------------------------------------------------
    encoding_session = tf.InteractiveSession()
    encoder, encoder_graph = load_encoder(
        encoding_session,
        layer,
        fov_size,
        seed,
        ffn_ckpt=ffn_ckpt,
        encoder_ckpt=encoder_ckpt,
    )

    # Embed input volumes ----------------------------------------------
    content_features = encoding_session.run(
        encoder.encoding, feed_dict={encoder.input_patches: content_volume}
    )
    style_features = encoding_session.run(
        encoder.encoding, feed_dict={encoder.input_patches: style_volume}
    )

    # Close this session, we're done with TF for now
    encoding_session.close()

    # Do feature transform --------------------------------------------
    # Whitening-coloring transform
    if method == 'wc':
        transformed_features = whitening_coloring_transform(
            content_features, style_features
        )

    # Histogram-matching transform
    elif method == 'hm':
        transformed_features = histogram_matching_transform(
            content_features, style_features
        )

    else:
        raise ValueError(f'Invalid method={method}.')

    # Load decoder ----------------------------------------------------
    decoding_session = tf.InteractiveSession()
    decoder, decoder_graph = load_decoder(
        decoding_session, layer, fov_size, decoder_ckpt
    )

    # Decode transformed features -------------------------------------
    decoded_transform = decoding_session.run(
        decoder.decoding,
        feed_dict={decoder.input_encoding: transformed_features},
    )

    # Write to output volume ------------------------------------------
    dx.writespec(outspec, decoded_transform)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Loving absl... I know I am fighting with it but w/e.
    flags.mark_flag_as_required('layer')
    flags.mark_flag_as_required('contentspec')
    flags.mark_flag_as_required('stylespec')
    flags.mark_flag_as_required('outspec')
    flags.mark_flag_as_required('decoder_ckpt')

    FLAGS = flags.FLAGS

    def _main(argv):
        # Deal with flags a bit
        assert FLAGS.encoder_ckpt or FLAGS.ffn_ckpt
        assert not (FLAGS.encoder_ckpt and FLAGS.ffn_ckpt)

        feature_transform_style_xfer(
            FLAGS.layer,
            FLAGS.contentspec,
            FLAGS.stylespec,
            FLAGS.outspec,
            FLAGS.decoder_ckpt,
            encoder_ckpt=FLAGS.encoder_ckpt,
            ffn_ckpt=FLAGS.ffn_ckpt,
            method=FLAGS.method,
        )

    app.run(_main)
