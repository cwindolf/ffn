# Flood-Filling Networks

This is a fork of [google/ffn](https://github.com/google/ffn) with some
details filled in for a more complete segmentation pipeline in a SLURM
environment. It also includes an implementation of the FFN authors'
[segmentation-enhanced CycleGAN](https://www.biorxiv.org/content/
10.1101/548081v1) for use in domain transfer problems with this repo's
FFN implementation. The below README will reflect the changes from the
original repo.

# About FFN

Flood-Filling Networks (FFNs) are a class of neural networks designed for
instance segmentation of complex and large shapes, particularly in volume
EM datasets of brain tissue.

For more details, see the related publications:

 * https://arxiv.org/abs/1611.00421
 * https://doi.org/10.1101/200675

This is most definitely not an official Google product.

# Installation

To install the necessary dependencies, run:

```shell
  pip install -r requirements.txt
```

The code has been tested on an Ubuntu 16.04.3 LTS system equipped with a
Tesla P100 GPU, and on a SLURM cluster with V100 GPUs.

Some scripts require that the `ffn` and `secgan` modules be installed
with

```shell
  python setup.py develop
```

or equivalent.

# Training

FFN networks can be trained with the `train.py` script, which expects a
TFRecord file of coordinates at which to sample data from input volumes.

In SLURM clusters, `slurm_train.py` should be used for distributed training
with asynchronous SGD. The API is similar to `train.py` and the data
preparation steps are still required.

## Preparing the training data

There are two scripts to generate training coordinate files for
a labeled dataset stored in HDF5 files: `compute_partitions.py` and
`build_coordinates.py`.

`compute_partitions.py` transforms the label volume into an intermediate
volume where the value of every voxel `A` corresponds to the quantized
fraction of voxels labeled identically to `A` within a subvolume of
radius `lom_radius` centered at `A`. `lom_radius` should normally be
set to `(fov_size // 2) + deltas` (where `fov_size` and `deltas` are
FFN model settings). Every such quantized fraction is called a *partition*.
Sample invocation:

```shell
  python compute_partitions.py \
    --input_volume third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --output_volume third_party/neuroproof_examples/validation_sample/af.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000
```

`build_coordinates.py` uses the partition volume from the previous step
to produce a TFRecord file of coordinates in which every partition is
represented approximately equally frequently. Sample invocation:

```shell
  python build_coordinates.py \
     --partition_volumes validation1:third_party/neuroproof_examples/validation_sample/af.h5:af \
     --coordinate_output third_party/neuroproof_examples/validation_sample/tf_record_file \
     --margin 24,24,24
```

## Sample data

We provide a sample coordinate file for the FIB-25 `validation1` volume
included in `third_party`. Due to its size, that file is hosted in
Google Cloud Storage. If you haven't used it before, you will need to
install the Google Cloud SDK and set it up with:

```shell
  gcloud auth application-default login
```

You will also need to create a local copy of the labels and image with:

```shell
  gsutil rsync -r -x ".*.gz" gs://ffn-flyem-fib25/ third_party/neuroproof_examples
```

## Running training

Once the coordinate files are ready, you can start training the FFN with:

```shell
  python train.py \
    --train_coords gs://ffn-flyem-fib25/validation_sample/fib_flyem_validation1_label_lom24_24_24_part14_wbbox_coords-*-of-00025.gz \
    --data_volumes validation1:third_party/neuroproof_examples/validation_sample/grayscale_maps.h5:raw \
    --label_volumes validation1:third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33
```

Note that both training and inference with the provided model are
computationally expensive processes. We recommend a GPU-equipped machine
for best results, particularly when using the FFN interactively in a Jupyter
notebook. Training the FFN as configured above requires a GPU with 12 GB of RAM.
You can reduce the batch size, model depth, `fov_size`, or number of features in
the convolutional layers to reduce the memory usage.

For multi-GPU or distributed training, see `slurm_train.py`.

# Inference

We provide two examples of how to run inference with a trained FFN model.
For a non-interactive setting, you can use the `run_inference.py` script:

```shell
  python run_inference.py \
    --inference_request="$(cat configs/inference_training_sample2.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'
```

which will segment the `training_sample2` volume and save the results in
the `results/fib25/training2` directory. Two files will be produced:
`seg-0_0_0.npz` and `seg-0_0_0.prob`. Both are in the `npz` format and
contain a segmentation map and quantized probability maps, respectively.
In Python, you can load the segmentation as follows:

```python
  from ffn.inference import storage
  seg, _ = storage.load_segmentation('results/fib25/training2', (0, 0, 0))
```

We provide sample segmentation results in `results/fib25/sample-training2.npz`.
For the training2 volume, segmentation takes about 7 min with a P100 GPU.

For an interactive setting, check out `ffn_inference_demo.ipynb`. This Jupyter
notebook shows how to segment a single object with an explicitly defined
seed and visualize the results while inference is running.

Both examples are configured to use a 3d convstack FFN model trained on the
`validation1` volume of the FIB-25 dataset from the FlyEM project at Janelia.

## Data- and model-parallel inference

To run inference on the same (small) region for many models at once on a
SLURM cluster, try `run_slurm_inference.py` and see `ffn/slurm/sinference.py`.
This is useful for checkpoint selection.

To run parallel inference on a large region by chunking it into overlapping
subvolumes, try `run_batch_inference.py`. This is helpful for scaling up to
larger regions than single-threaded inference can support. It supports
parallelism within a single GPU and across multiple GPUs by a simple work
division strategy.

## Combining multiple inferences

After running multiple inferences of the same volume, or after running an
inference over a region that has been chunked into many subvolumes, you
can combine those inferences using `run_consensus.py`.

For instance, say that you have run `M` inferences of the same region (for
example, forward inferences with `PolicyPeaks` and reverse inferences with
`PolicyInvertOrigins`, or for another example, multiple inferences of the
same region with multiple models). Also, you may have chunked that region
into `N` overlapping cubes with `run_batch_inference.py`.

In that situation, `run_consensus.py` will compute the "meet" (i.e. it will
combine all of the splits) over the `M` inferences for each subvolume. Then,
the `N` subvolumes will be combined by a simple procedure into one large
volume, which will be stored in an HDF5 file.

# Affinities

To facilitate proofreading of a large volume, it may be necessary to first
oversegment that volume (possibly by combining all splits from multiple
inferences with `run_consensus.py`), and then compute "affinities" between
the supervoxels in the oversegmentation. Those supervoxels can be used to
generate an initial automatic merge, which can be proofread more easily
by using a tool like [neuclease](https://github.com/janelia-flyem/neuclease).

Affinities can be generated in an FFN-guided fashion using the script
`run_resegmentation.py`. That script allows you to compute affinities and
optionally compute and post an automatic merge to a DVID server. It uses
the FFN repo's "resegmentation" infrastructure to do this, see the script
and the `ffn/inference/resegmentation*` files for more info.
