
/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/vas_transformer_nofeat.yaml -t True --gpus 0,1,2,3 model.params.first_stage_config.params.ckpt_path=/modelblob/users/v-chengw/models/specvqgan/2022-08-15T15-03-10_vas_codebook/checkpoints/last.ckpt --logdir /modelblob/users/v-chengw/models/specvqgan/
