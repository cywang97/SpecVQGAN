
/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/vggsound_transformer_new.yaml -t True --gpus 0,1,2,3,4,5,6,7 model.params.first_stage_config.params.ckpt_path=/modelblob/users/v-chengw/models/specvqgan/2022-08-15T17-21-46_vggsound_codebook/checkpoints/last.ckpt --logdir /modelblob/users/v-chengw/models/specvqgan/
