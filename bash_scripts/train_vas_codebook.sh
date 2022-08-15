strings=("dog" "fireworks" "drum" "baby" "gun" "sneeze" "cough" "hammer")
mkdir -p ./data/vas/features/
for class in "${strings[@]}"; do
	cp -r /datablob/users/v-chengw/data/specvqgan/downloaded_features/vas/${class} ./data/vas/features/
done

/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/vas_codebook.yaml -t True --gpus 0,1,2,3 --logdir /modelblob/users/v-chengw/models/specvqgan/
