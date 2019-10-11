SRUNFLAGS=-p gpu --gres=gpu:1 --constraint=v100 --output="%x_%N_%j.out" --error="%x_%N_%j.err"
CKPTDIR=/mnt/ceph/data/neuro/wasp_em/xfer/ckpt
FFNCKPT=/mnt/ceph/data/neuro/wasp_em/ffndat/ckpt/dtwoc_190610_postclean/model.ckpt-120057214
BMED=/mnt/home/cwindolf/data/raw/bmed_zyxl_3300_7300_3000_600.h5:raw
HEAD2MEDGHM=~/data/raw/head2_med_ghm_zyxl_13800_6000_3600_500.h5:raw
TRAINDECFLAGS=--batch_size 4 --optimizer adam --ffn_ckpt $(FFNCKPT) --volume_spec $(BMED) --max_steps 50000
DECCKPT=model.ckpt-70004
TRAINENCFLAGS=--volume_spec $(HEAD2MEDGHM) --max_steps 100000

.PHONY: train_decoders
train_decoders: xfer3 xfer5 xfer7
	@echo "Trained some decoders."

.PHONY: xfer3
xfer3:
	srun $(SRUNFLAGS) --job-name xfer3 \
		python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer3/ \
		--layer 3 \


.PHONY: xfer5
xfer5:
	srun $(SRUNFLAGS) --job-name xfer5 \
	python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer5/ \
		--layer 5 \

.PHONY: xfer7
xfer7:
	srun $(SRUNFLAGS) --job-name xfer7 \
	python train_decoder.py \
		$(TRAINDECFLAGS) \
		--train_dir $(CKPTDIR)/bmed_layer7/ \
		--layer 7 \

.PHONY: encdechead2inf
encdechead2inf:
	python segval/slurm_infer.py \
		--infreqs="$(echo /mnt/ceph/data/neuro/wasp_em/ffndat/cfg/encdec*_head2_medghm.pbtxt)" \
		--bboxes $CEPHFFN/cfg/5box.pbtxt \
		--where cluster

.PHONY: train_encoders
train_all_encoders: train_encoders train_encoders_lambdaflip

.PHONY: train_encoders
train_encoders: xferenc3 xferenc3ffn xferenc5 xferenc5ffn xferenc7 xferenc7ffn

.PHONY: xferenc3
xferenc3:
	srun $(SRUNFLAGS) --job-name xferenc3 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer3/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc3ffn
xferenc3ffn:
	srun $(SRUNFLAGS) --job-name xferenc3ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer3_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc5
xferenc5:
	srun $(SRUNFLAGS) --job-name xferenc5 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer5/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc5ffn
xferenc5ffn:
	srun $(SRUNFLAGS) --job-name xferenc5ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer5_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc7
xferenc7:
	srun $(SRUNFLAGS) --job-name xferenc7 \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer7/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)

.PHONY: xferenc7ffn
xferenc7ffn:
	srun $(SRUNFLAGS) --job-name xferenc7ffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_layer7_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)


.PHONY: train_encoders_lambdaflip
train_encoders_lambdaflip: xferenc3ll xferenc3llffn xferenc5ll xferenc5llffn xferenc7ll xferenc7llffn

.PHONY: xferenc3ll
xferenc3ll:
	srun $(SRUNFLAGS) --job-name xferenc3ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer3/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc3llffn
xferenc3llffn:
	srun $(SRUNFLAGS) --job-name xferenc3llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 3 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer3_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer3/$(DECCKPT)

.PHONY: xferenc5ll
xferenc5ll:
	srun $(SRUNFLAGS) --job-name xferenc5ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer5/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc5llffn
xferenc5llffn:
	srun $(SRUNFLAGS) --job-name xferenc5llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 5 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer5_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer5/$(DECCKPT)

.PHONY: xferenc7ll
xferenc7ll:
	srun $(SRUNFLAGS) --job-name xferenc7ll \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer7/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)

.PHONY: xferenc7llffn
xferenc7llffn:
	srun $(SRUNFLAGS) --job-name xferenc7llffn \
		python train_encoder.py $(TRAINENCFLAGS) \
		--layer 7 \
		--pixel_loss_lambda 1.0 \
		--encoding_loss_lambda 1e-3 \
		--ffn_ckpt $(FFNCKPT) \
		--train_dir $(CKPTDIR)/enc_head2medghm_lamflip_layer7_from_ffn/ \
		--decoder_ckpt $(CKPTDIR)/bmed_layer7/$(DECCKPT)
