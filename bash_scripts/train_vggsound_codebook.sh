cp /datablob/users/v-chengw/data/specvqgan/downloaded_features/vggsound/*.tar ./data/vggsound/
for i in $(seq 1 9); do
	tar xf data/vggsound/melspec_10s_22050hz_0${i}.tar -C data/vggsound/
	rm data/vggsound/melspec_10s_22050hz_0${i}.tar
done
for i in $(seq 10 64); do
	tar xf data/vggsound/melspec_10s_22050hz_${i}.tar -C data/vggsound/
	rm data/vggsound/melspec_10s_22050hz_${i}.tar
done


/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/vggsound_codebook.yaml -t True --gpus 0,1,2,3,4,5,6,7 --logdir /modelblob/users/v-chengw/models/specvqgan/
