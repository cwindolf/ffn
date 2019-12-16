import numpy as np
import tensorflow as tf
import preprocessing.data_util as dx
from models.secgan import SECGAN


def secgan_infer(
    content_volume_spec,
    output_spec,
    checkpoint_path,
    ffn_fov_size=33,
    generator_depth=8,
    generator_channels=32,
    generator_margin=16,
    nchunks=1,
    F_or_G='F',
):
    '''Run the generator F to map an unabeled volume to a labeled-ish volume
    '''
    # Load data
    content_volume = dx.loadspec(content_volume_spec)

    # Process data
    content_volume = content_volume.astype(np.float32)
    content_volume /= 127.5
    content_volume -= 1.0

    # Make chunks
    # They need to overlap according to generator_margin
    # Chunksize is kind of a misnomer.
    # First and last chunks will have size chunksize + gen_margin,
    # all others chunksize + 2 * genmargin,
    # Unless the first and last chunks are the same in which
    # case they'll just be the whole volume.
    # And of course there might be an off-by-one on the last slice
    # to deal with integer division.
    chunksize = content_volume.shape[0] // nchunks
    chunksizes = [chunksize] * (nchunks - 1)
    chunksizes.append(content_volume.shape[0] - sum(chunksizes))
    chunks = []
    for i, c in enumerate(chunksizes):
        if i == 0:
            lo = 0
        else:
            lo = i * c - generator_margin

        if i == nchunks - 1:
            hi = content_volume.shape[0]
        else:
            hi = (i + 1) * c + generator_margin

        assert 0 <= lo <= content_volume.shape[0] - chunksize
        assert chunksize <= hi <= content_volume.shape[0]

        chunks.append(content_volume[lo:hi])

    # Init model
    generator = SECGAN(
        batch_size=1,
        ffn_fov_shape=(ffn_fov_size, ffn_fov_size, ffn_fov_size),
        generator_depth=generator_depth,
        generator_channels=generator_channels,
        inference_ckpt=checkpoint_path,
        seg_enhanced=False,
    )

    # TF world
    with tf.Graph().as_default():
        if F_or_G == 'F':
            generator.define_F_graph(chunks[0].shape)
        elif F_or_G == 'G':
            generator.define_G_graph(chunks[0].shape)
        else:
            raise ValueError(
                f'F or G? Like, I only ask because you put "{F_or_G}"...'
            )

        xfer_chunks = []

        with tf.Session() as sess:
            sess.run(generator.inf_init_op, feed_dict=generator.inf_init_fd)

            for chunk in chunks:
                xfer_chunk = sess.run(
                    generator.xfer_output,
                    feed_dict={generator.xfer_input: chunk[None, ..., None]},
                )
                xfer_chunks.append(xfer_chunk.squeeze())

    # Post process
    # For now, let's just leave this centered and float typed.
    # But multiply by 127.5 since that will be divided out later
    # by (my fork of) the FFN inference script, I think, haha.
    # IDK, depends on the inference request.
    xfer_volume = np.concatenate(xfer_chunks)
    assert all(
        np.array(xfer_volume.shape)
        == (np.array(content_volume.shape) - 2 * generator_margin)
    )
    # xfer_volume += 1.0
    # xfer_volume *= 127.5

    # Write output
    dx.writespec(
        output_spec,
        xfer_volume,
        h5_attrs={
            'checkpoint_path': checkpoint_path,
            'content_volume_spec': content_volume_spec,
        },
    )


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------

    from absl import app
    from absl import flags

    flags.DEFINE_string('content_volume_spec', None, '')
    flags.DEFINE_string('output_spec', None, '')
    flags.DEFINE_string('checkpoint_path', None, '')

    flags.DEFINE_string('F_or_G', 'F', '')
    flags.DEFINE_integer('ffn_fov_size', 33, '')
    flags.DEFINE_integer('generator_depth', 8, '')
    flags.DEFINE_integer('generator_channels', 32, '')
    flags.DEFINE_integer('generator_margin', 16, '')
    flags.DEFINE_integer('nchunks', 1, '')

    FLAGS = flags.FLAGS

    def main(argv):
        secgan_infer(
            FLAGS.content_volume_spec,
            FLAGS.output_spec,
            FLAGS.checkpoint_path,
            ffn_fov_size=FLAGS.ffn_fov_size,
            generator_depth=FLAGS.generator_depth,
            generator_channels=FLAGS.generator_channels,
            generator_margin=FLAGS.generator_margin,
            F_or_G=FLAGS.F_or_G,
            nchunks=FLAGS.nchunks,
        )

    app.run(main)
