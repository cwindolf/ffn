import numpy as np
import scipy.linalg as la
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import MinCovDet
import preprocessing.data_util as dx
import models
from training import inputs
from util import whiten
import logging
from absl import flags
from absl import app


# ------------------------------- lib ---------------------------------


# TensorFlow business -------------------------------------------------


def load_encoder(
    layer, fov_size, batch_size=1, ffn_ckpt=None, encoder_ckpt=None
):
    # Load FFN weights ------------------------------------------------
    # Hooking graphs together... This bit loads up weights.
    if ffn_ckpt:
        assert not encoder_ckpt
        encoder = models.ConvStack3DEncoder.from_ffn_ckpt(
            ffn_ckpt=ffn_ckpt,
            fov_size=fov_size,
            depth=layer,
            batch_size=batch_size,
            seed_as_placeholder=True,
        )

    # Model graph -----------------------------------------------------
    encoder_graph = tf.Graph()
    with encoder_graph.as_default():
        # Init encoder ------------------------------------------------
        if not ffn_ckpt:
            assert encoder_ckpt
            encoder = models.ConvStack3DEncoder(
                fov_size=fov_size,
                depth=layer,
                batch_size=batch_size,
                seed_as_placeholder=True,
            )
        encoder.define_tf_graph()

        # Make init fn
        if ffn_ckpt:
            einit_op = tf.initializers.variables(encoder.vars)

            def einit_fn(session):
                session.run(einit_op)

        else:

            def einit_fn(session):
                encoder.saver.restore(session, encoder_ckpt)

    return encoder, encoder_graph, einit_fn


def load_decoder(layer, fov_size, decoder_ckpt, batch_size=1):
    decoder_graph = tf.Graph()
    with decoder_graph.as_default():
        # Init decoder
        decoder = models.ConvStack3DDecoder(
            fov_size=fov_size,
            depth=layer,
            batch_size=batch_size,
            for_training=False,
        )
        decoder.define_tf_graph()

        # Restore
        def dinit_fn(session):
            decoder.saver.restore(session, decoder_ckpt)

    return decoder, decoder_graph, dinit_fn


# Feature transforms --------------------------------------------------


def whitening_coloring_transform(
    content_features, style_features, whitening='zca', fudge=1e-8
):
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
    nfeatures = orig_content_shape[3]
    assert style_features.shape[3] == nfeatures
    content_features = content_features.transpose(3, 0, 1, 2)
    content_features = content_features.reshape(nfeatures, -1)
    content_features = content_features.astype(np.float64)
    # sqrt_N = np.sqrt(nsamples)
    style_features = style_features.transpose(3, 0, 1, 2)
    style_features = style_features.reshape(nfeatures, -1)
    style_features = style_features.astype(np.float64)

    # Whitening transform ---------------------------------------------
    # Center
    content_feature_means = content_features.mean(axis=1, keepdims=True)
    cfms = ' '.join(map(str, content_feature_means.flat))
    logging.info(f'Content feature means:\n{cfms}')
    content_features -= content_feature_means

    # Compute whitening
    whitened_features = whiten.whiten(
        content_features, method=whitening, assume_centered=True
    )

    # Coloring transform ----------------------------------------------
    # Center
    style_feature_means = style_features.mean(
        axis=1, keepdims=True, dtype=np.float64
    )
    style_feature_means = style_feature_means.astype(style_features.dtype)
    sfms = ' '.join(map(str, style_feature_means.flat))
    logging.info(f'Style feature means:\n{sfms}')
    style_features -= style_feature_means

    # Compute coloring
    colorizer = whiten.coloring_matrix(
        style_features, method=whitening, assume_centered=True
    )
    wct_features = colorizer @ whitened_features + style_feature_means

    # Shape back to space x features
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
    nfeatures = orig_content_shape[3]
    assert style_features.shape[3] == nfeatures
    content_features = content_features.transpose(3, 0, 1, 2)
    content_features = content_features.reshape(nfeatures, -1)
    style_features = style_features.transpose(3, 0, 1, 2)
    style_features = style_features.reshape(nfeatures, -1)

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
    whitening='zca',
    image_mean=128.0,
    image_stddev=33.0,
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
    # Load data -------------------------------------------------------
    logging.info('Loading data')

    content_volume = dx.loadspec(contentspec)
    style_volume = dx.loadspec(stylespec)
    if style_volume.shape != content_volume.shape:
        raise ValueError(
            'Style and content volumes were different shapes. This case '
            'can be supported but it will take a little work. Maybe not '
            'too much work actually.'
        )
    fov_size = content_volume.shape

    # Init seed
    seed = inputs.fixed_seed_batch(1, fov_size, 0.5, 0.5)

    # Some stats on input
    cstats = (
        content_volume.min(),
        content_volume.mean(),
        content_volume.max(),
    )
    logging.info(f'Before rescaling content image, (min,mean,max):{cstats}.')
    sstats = (style_volume.min(), style_volume.mean(), style_volume.max())
    logging.info(f'Before rescaling style image, (min,mean,max):{sstats}.')

    # Center and scale per FFN
    content_volume = (
        content_volume.astype(np.float32) - image_mean
    ) / image_stddev
    style_volume = (
        style_volume.astype(np.float32) - image_mean
    ) / image_stddev

    # Add batch + feature dim
    content_volume = content_volume[None, ..., None]
    style_volume = style_volume[None, ..., None]

    # Load encoder ----------------------------------------------------
    logging.info('Loading encoder')

    encoder, encoder_graph, einit_fn = load_encoder(
        layer, fov_size, ffn_ckpt=ffn_ckpt, encoder_ckpt=encoder_ckpt
    )

    # Embed input volumes ----------------------------------------------
    with tf.Session(graph=encoder_graph) as sess:
        einit_fn(sess)

        logging.info('Getting content features')
        content_features = sess.run(
            encoder.encoding,
            feed_dict={
                encoder.input_patches: content_volume,
                encoder.input_seed: seed,
            },
        )

        logging.info('Getting style features')
        style_features = sess.run(
            encoder.encoding,
            feed_dict={
                encoder.input_patches: style_volume,
                encoder.input_seed: seed,
            },
        )

    # Unbatch
    content_features = content_features.squeeze()
    style_features = style_features.squeeze()

    # Do feature transform --------------------------------------------
    # Whitening-coloring transform
    if method.startswith('wc'):
        logging.info('Applying WCT')
        transformed_features = whitening_coloring_transform(
            content_features, style_features, whitening=whitening
        )

    # Histogram-matching transform
    elif method.startswith('hm'):
        logging.info('Applying HMT')
        transformed_features = histogram_matching_transform(
            content_features, style_features
        )

    else:
        raise ValueError(f'Invalid method={method}.')

    # Report numerical issues
    nans = np.isnan(transformed_features).any()
    logging.info(
        f'Did the feature transform produce NaNs? {"yes" if nans else "no"}'
    )

    # Batch again
    transformed_features = transformed_features[None, ...]

    # Load decoder ----------------------------------------------------
    logging.info(f'Loading decoder from {decoder_ckpt}')

    decoder, decoder_graph, dinit_fn = load_decoder(
        layer, fov_size, decoder_ckpt
    )

    # Decode transformed features -------------------------------------
    with tf.Session(graph=decoder_graph) as sess:
        dinit_fn(sess)

        logging.info('Decoding transformed features')
        transform_result = sess.run(
            decoder.decoding,
            feed_dict={decoder.input_encoding: transformed_features},
        )

    # The final unbatching
    transform_result = transform_result.squeeze()

    # Un-rescale
    transform_result *= image_stddev
    transform_result += image_mean
    dstats = (
        transform_result.min(),
        transform_result.mean(),
        transform_result.max(),
    )
    logging.info(f'After rescaling the decoding, (min,mean,max):{dstats}.')

    # Write to output volume ------------------------------------------
    logging.info(f'Saving result to {outspec}')
    dx.writespec(outspec, transform_result)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Flags -----------------------------------------------------------

    # Loving absl... I know I am fighting with it but w/e.
    flags.DEFINE_enum(
        'method',
        'wc',
        ['wc', 'wct', 'hm', 'hmt'],
        'Whitening-coloring or histogram matching',
    )
    flags.DEFINE_string(
        'contentspec', None, 'Input datspec, the content source.'
    )
    flags.DEFINE_string('stylespec', None, 'Target style datspec.')
    flags.DEFINE_string('outspec', None, 'Write output here.')
    flags.DEFINE_string('whitening', 'zca', 'What whitening?')

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

    flags.mark_flag_as_required('layer')
    flags.mark_flag_as_required('contentspec')
    flags.mark_flag_as_required('stylespec')
    flags.mark_flag_as_required('outspec')
    flags.mark_flag_as_required('decoder_ckpt')

    FLAGS = flags.FLAGS

    # Main ------------------------------------------------------------

    def _main(argv):
        # Deal with flags a bit
        assert FLAGS.encoder_ckpt or FLAGS.ffn_ckpt
        assert not (FLAGS.encoder_ckpt and FLAGS.ffn_ckpt)

        logging.basicConfig(level=logging.DEBUG)

        feature_transform_style_xfer(
            FLAGS.layer,
            FLAGS.contentspec,
            FLAGS.stylespec,
            FLAGS.outspec,
            FLAGS.decoder_ckpt,
            encoder_ckpt=FLAGS.encoder_ckpt,
            ffn_ckpt=FLAGS.ffn_ckpt,
            method=FLAGS.method,
            whitening=FLAGS.whitening,
        )

    app.run(_main)
