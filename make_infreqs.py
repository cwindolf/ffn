import argparse
import glob
import os
import preprocessing.data_util as dx


swapped_template = """image {{
  hdf5: "{datspec}"
}}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/mnt/ceph/data/neuro/wasp_em/ffndat/ckpt/dtwoc_190610_postclean/model.ckpt-120057214"
model_name: "swapped_encoder_convstack_3d.SwappedEncoderFFNModel"
model_args: "{{\\"encoder_ckpt\\": \\"{ckptfolder}/model.ckpt-{ckptnum}\\", \\"fov_size\\": [33, 33, 33], \\"deltas\\": [8, 8, 8], \\"depth\\": 12, \\"layer\\": {layer}}}"
segmentation_output_dir: "{outdir}"
inference_options {{
  init_activation: 0.95
  pad_value: 0.5
  move_threshold: 0.6
  min_boundary_dist {{ x: 1 y: 1 z: 1 }}
  segment_threshold: 0.6
  min_segment_size: 1000
}}
"""

ap = argparse.ArgumentParser()

ap.add_argument('datspec', type=str)
ap.add_argument('run_name', type=str)
ap.add_argument('ckptfolders', nargs='+')
ap.add_argument(
    '--infreqsavedir', default='/mnt/ceph/data/neuro/wasp_em/xfer/cfg/'
)
ap.add_argument(
    '--ckptbasedir', default='/mnt/ceph/data/neuro/wasp_em/xfer/ckpt/'
)
ap.add_argument(
    '--infbasedir', default='/mnt/ceph/data/neuro/wasp_em/xfer/inf/'
)

args = ap.parse_args()

if not os.path.exists(os.path.join(args.infbasedir, args.run_name)):
    os.mkdir(os.path.join(args.infbasedir, args.run_name))
if not os.path.exists(os.path.join(args.infreqsavedir, args.run_name)):
    os.mkdir(os.path.join(args.infreqsavedir, args.run_name))

for ckptfolder in args.ckptfolders:
    if not ckptfolder.startswith('/mnt/ceph'):
        ckptfolder = os.path.join(args.infreqsavedir, ckptfolder)

    mname = ckptfolder.split('/')[-1]
    layer = dx.extract_ints(mname)

    # Find latest checkpoint
    latestckpt = max(
        map(
            dx.extract_ints, glob.glob(os.path.join(ckptfolder, 'model.ckpt-*'))
        )
    )

    # Make inference folder
    inffoldername = f'{mname}_{latestckpt}'
    outdir = os.path.join(args.infbasedir, args.run_name, inffoldername)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if os.path.exists(os.path.join(outdir, '0')):
        print(f'There is already stuff in {outdir}')

    # Write infreq
    with open(
        os.path.join(
            args.infreqsavedir, args.run_name, f'{mname}_{latestckpt}.pbtxt'
        ),
        'w',
    ) as pb:
        pb.write(
            swapped_template.format(
                datspec=args.datspec,
                ckptfolder=ckptfolder,
                ckptnum=latestckpt,
                outdir=outdir,
                layer=layer,
            )
        )
