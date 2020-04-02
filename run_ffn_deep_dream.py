"""
This is not exactly a deep dream. Not sure what to call it.
"""
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.morphology as morpho
from skimage.filters import gaussian

from ffn.training.models import convstack_3d
from secgan.models import convstacktools

# ------------------------------ library ------------------------------


class bwdrawer:
    """Draw one pixel at a time on a black and white canvas."""

    def __init__(self, w, h):
        self.pen = False
        self.canvas = np.zeros((h, w), dtype=np.uint8)
        self.fig, self.ax = plt.subplots(1, 1)
        self.im = None
        self.hardreset()
        self.fig.canvas.mpl_connect("button_press_event", self.down)
        self.fig.canvas.mpl_connect("button_release_event", self.up)
        self.fig.canvas.mpl_connect("motion_notify_event", self.move)
        plt.show(block=True)

    def hardreset(self):
        self.im = self.ax.imshow(self.canvas, cmap="binary", vmin=0, vmax=1)
        self.fig.canvas.draw()

    def update(self):
        self.im.set_data(self.canvas)
        self.fig.canvas.draw()

    def down(self, event):
        print(event)
        self.pen = True
        y, x = int(np.floor(event.ydata)), int(np.floor(event.xdata))
        self.canvas[y, x] = 1
        self.update()

    def up(self, _):
        self.pen = False

    def move(self, event):
        if self.pen:
            y, x = int(np.floor(event.ydata)), int(np.floor(event.xdata))
            self.canvas[y, x] = 1
            self.update()


if __name__ == "__main__":
    # ----------------------------- args ------------------------------
    ap = argparse.ArgumentParser()

    size_g = ap.add_argument_group(title="Size")
    size_g.add_argument("w", type=int, default=33, nargs="?")
    size_g.add_argument("h", type=int, default=33, nargs="?")
    size_g.add_argument("d", type=int, default=33, nargs="?")

    ap.add_argument("--ckpt", help="Path to FFN checkpoint.")
    ap.add_argument(
        "--depth", type=int, default=12, help="Number of FFN layers."
    )

    ap.add_argument("--neuron-radius", type=int, default=4)
    ap.add_argument("--blur-radius", type=int, default=0.5)

    args = ap.parse_args()

    # ------------------------- make dd input -------------------------
    # Get neuron skeleton by drawing input
    draw = bwdrawer(args.w, args.h)

    # Turn skeleton neuron blob
    skeleton = np.zeros((args.d, args.h, args.w), dtype=np.float32)
    skeleton[args.d // 2] = draw.canvas
    neuron = morpho.binary_dilation(
        skeleton, selem=morpho.ball(args.neuron_radius)
    )
    bluron = gaussian(neuron, sigma=args.blur_radius)

    # Show what we got
    plt.imshow(bluron[args.d // 2], cmap="gray")
    plt.show(block=True)

    # -------------------------- dd tf setup --------------------------
    # We fix `bluron` as the input mask and output, and descend on
    # FFN's loss wrt its image input, passed in as a variable.

    # Load up the weights from the ffn checkpoint, then discard all
    # of the loading infrastructure.
    ffn_loading_graph = tf.Graph()
    with ffn_loading_graph.as_default():
        ffn = convstack_3d.ConvStack3DFFNModel(
            # These parameters don't matter right now since we're not
            # going to use this object except as a weights loader.
            fov_size=[33, 33, 33],
            deltas=[1, 1, 1],
            batch_size=1,
            # The depth does matter tho!
            depth=args.depth,
        )
        # Since ffn.labels == None, this will not set up training graph
        ffn.define_tf_graph()

        trainable_names = [v.op.name for v in tf.trainable_variables()]

        with tf.Session(graph=ffn_loading_graph) as sess:
            ffn.saver.restore(sess, args.ckpt)
            weights = dict(
                zip(trainable_names, sess.run(tf.trainable_variables()))
            )
    # Just to be explicit about not using this, for readers of this code.
    del ffn, ffn_loading_graph

    # Instead of using FFN, make a fixed version of the FFN graph
    with tf.Graph().as_default():
        dd_image = tf.Variable(
            initial_value=np.random.normal(
                scale=0.1, size=(1, *bluron.shape, 1)
            )
        )
        init_op = tf.variables_initializer([dd_image])
        mask = tf.constant(bluron[None, ..., None])
        net = tf.concat([dd_image, mask], axis=4)
        logits = convstacktools.fixed_convstack_3d(
            net, weights, depth=args.depth
        )
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=mask)
        )
        opt = tf.train.AdamOptimizer()
        dd_op = opt.minimize(loss, var_list=[dd_image])
        dd_slice = dd_image[0, args.d // 2, :, :, 0]

        # deep dream loop
        with tf.Session() as sess:
            sess.run(init_op)
            while True:
                dd = sess.run(dd_slice)
                plt.imshow(dd)
                plt.show(block=True)
                for _ in range(10):
                    _ = sess.run(dd_op)
