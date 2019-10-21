CEPHWASP=/mnt/ceph/data/neuro/wasp_em
CEPHXFER=$(CEPHWASP)/xfer
LOGDIR=$(CEPHXFER)/logs
REDIRFLAGS=--output="$(LOGDIR)/%x_%N_%j.out" --error="$(LOGDIR)/%x_%N_%j.err"
CPUSRUNFLAGS=-p ccb -N 1 --exclusive $(REDIRFLAGS)
GPUSRUNFLAGS=-p gpu $(REDIRFLAGS) --gres=gpu:1 --constraint=v100
CKPTDIR=$(CEPHXFER)/ckpt
FFNCKPT=$(CEPHWASP)/ffndat/ckpt/dtwoc_190610_postclean/model.ckpt-120057214
BMED=/mnt/home/cwindolf/data/raw/bmed_zyxl_3200_7200_2900_900.h5:raw
HEAD2MEDGHM=~/data/raw/head2_med_ghm_zyxl_13800_6000_3600_500.h5:raw


# ---------------------------------------------------------------------
# -------------------- inference batching targets ---------------------
# ---------------------------------------------------------------------


# ----------------------- encoder->decoder->ffn -----------------------

.PHONY: encdechead2inf
encdechead2inf:
	python segval/slurm_infer.py \
		--infreqs="$(echo /mnt/ceph/data/neuro/wasp_em/ffndat/cfg/encdec*_head2_medghm.pbtxt)" \
		--bboxes $(CEPHFFN)/cfg/5box.pbtxt \
		--where cluster


# -------------- encoder->decoder reconstruction checks ---------------

RECDIR=$(CEPHXFER)/rec
# RECFLAGS=--inspec $(HEAD2MEDGHM) --seed_type random
RECFLAGS=--inspec $(HEAD2MEDGHM)

.PHONY: reconstruction_grid
reconstruction_grid: ffn_reconstructions enc_reconstructions
	@echo "Reconstructing the grid."

.PHONY: ffn_reconstructions
ffn_reconstructions: ffn2reco ffn3reco ffn5reco ffn7reco
	@echo "Reconstructing from the FFN encoders."

.PHONY: ffn2reco
ffn2reco:
	mkdir -p $(RECDIR)/ffn2reco
	srun $(CPUSRUNFLAGS) --job-name ffn2reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/ffn2reco/ffn2reco.h5:raw \
			--layer 2 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer2/model.ckpt-100002 \
			--ffn_ckpt $(FFNCKPT)

.PHONY: ffn3reco
ffn3reco:
	mkdir -p $(RECDIR)/ffn3reco
	srun $(CPUSRUNFLAGS) --job-name ffn3reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/ffn3reco/ffn3reco.h5:raw \
			--layer 3 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer3/model.ckpt-100002 \
			--ffn_ckpt $(FFNCKPT)

.PHONY: ffn5reco
ffn5reco:
	mkdir -p $(RECDIR)/ffn5reco
	srun $(CPUSRUNFLAGS) --job-name ffn5reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/ffn5reco/ffn5reco.h5:raw \
			--layer 5 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer5/model.ckpt-100002 \
			--ffn_ckpt $(FFNCKPT)

.PHONY: ffn7reco
ffn7reco:
	mkdir -p $(RECDIR)/ffn7reco
	srun $(CPUSRUNFLAGS) --job-name ffn7reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/ffn7reco/ffn7reco.h5:raw \
			--layer 7 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer7/model.ckpt-100002 \
			--ffn_ckpt $(FFNCKPT)

.PHONY: enc_reconstructions
enc_reconstructions: enc3reco enc5reco enc7reco
	@echo "Reconstructing from the trained encoders."

.PHONY: enc3reco
enc3reco:
	mkdir -p $(RECDIR)/enc3reco
	srun $(CPUSRUNFLAGS) --job-name enc3reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/enc3reco/enc3reco.h5:raw \
			--layer 3 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer3/model.ckpt-70004 \
			--encoder_ckpt $(CKPTDIR)/enc_head2medghm_lamflip_layer3_from_ffn/model.ckpt-100002

.PHONY: enc5reco
enc5reco:
	mkdir -p $(RECDIR)/enc5reco
	srun $(CPUSRUNFLAGS) --job-name enc5reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/enc5reco/enc5reco.h5:raw \
			--layer 5 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer5/model.ckpt-70004 \
			--encoder_ckpt $(CKPTDIR)/enc_head2medghm_lamflip_layer5_from_ffn/model.ckpt-100002

.PHONY: enc7reco
enc7reco:
	mkdir -p $(RECDIR)/enc7reco
	srun $(CPUSRUNFLAGS) --job-name enc7reco \
		python reconstruct_volume.py \
			$(RECFLAGS) \
			--outspec $(RECDIR)/enc7reco/enc7reco.h5:raw \
			--layer 7 \
			--decoder_ckpt $(CKPTDIR)/bmed_layer7/model.ckpt-70004 \
			--encoder_ckpt $(CKPTDIR)/enc_head2medghm_lamflip_layer7_from_ffn/model.ckpt-100002


# ---------------------------------------------------------------------
# ---------------------- model training targets -----------------------
# ---------------------------------------------------------------------


# ------------------------- decoder training --------------------------

TRAINDECFLAGS=--batch_size 4 --optimizer adam --ffn_ckpt $(FFNCKPT) --volume_spec $(BMED) --max_steps 100000


.PHONY: train_decoders
train_decoders: xfer2 xfer3 xfer5 xfer7
	@echo "Trained some decoders."

.PHONY: xfer1
xfer1:
	srun $(GPUSRUNFLAGS) --job-name xfer1 \
		python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer1/ \
		--layer 1

.PHONY: xfer2
xfer2:
	srun $(GPUSRUNFLAGS) --job-name xfer2 \
		python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer2/ \
		--layer 2

.PHONY: xfer3
xfer3:
	srun $(GPUSRUNFLAGS) --job-name xfer3 \
		python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer3/ \
		--layer 3

.PHONY: xfer5
xfer5:
	srun $(GPUSRUNFLAGS) --job-name xfer5 \
	python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer5/ \
		--layer 5

.PHONY: xfer7
xfer7:
	srun $(GPUSRUNFLAGS) --job-name xfer7 \
	python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer7/ \
		--layer 7


# ------------------------- encoder training --------------------------

DECCKPT=model.ckpt-70004
TRAINENCFLAGS=--volume_spec $(HEAD2MEDGHM) --max_steps 100000

.PHONY: train_all_encoders
train_all_encoders: train_encoders_noflip train_encoders_lambdaflip

.PHONY: train_encoders_noflip
train_encoders_noflip: xferenc3 xferenc3ffn xferenc5 xferenc5ffn xferenc7 xferenc7ffn

.PHONY: xferenc3
xferenc3:
	srun $(GPUSRUNFLAGS) --job-name xferenc3 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer3/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc3ffn
xferenc3ffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc3ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer3_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc5
xferenc5:
	srun $(GPUSRUNFLAGS) --job-name xferenc5 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer5/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc5ffn
xferenc5ffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc5ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer5_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc7
xferenc7:
	srun $(GPUSRUNFLAGS) --job-name xferenc7 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer7/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)

.PHONY: xferenc7ffn
xferenc7ffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc7ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer7_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)


.PHONY: train_encoders_lambdaflip
train_encoders_lambdaflip: xferenc3ll xferenc3llffn xferenc5ll xferenc5llffn xferenc7ll xferenc7llffn

.PHONY: xferenc3ll
xferenc3ll:
	srun $(GPUSRUNFLAGS) --job-name xferenc3ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer3/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc3llffn
xferenc3llffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc3llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer3_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc5ll
xferenc5ll:
	srun $(GPUSRUNFLAGS) --job-name xferenc5ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer5/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc5llffn
xferenc5llffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc5llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer5_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc7ll
xferenc7ll:
	srun $(GPUSRUNFLAGS) --job-name xferenc7ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer7/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)

.PHONY: xferenc7llffn
xferenc7llffn:
	srun $(GPUSRUNFLAGS) --job-name xferenc7llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer7_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)
