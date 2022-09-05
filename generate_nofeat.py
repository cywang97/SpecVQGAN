import os
import time
import pdb
import numpy as np
import soundfile as sf

from pathlib import Path

import IPython.display as display_audio
import soundfile
import torch
from IPython import display
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid
from tqdm import tqdm

from feature_extraction.demo_utils import (ExtractResNet50, check_video_for_audio,
                                            extract_melspectrogram, load_model,
                                            show_grid, trim_video)
from sample_visualization import (all_attention_to_st, get_class_preditions,
                                last_attention_to_st, spec_to_audio_to_st,
                                tensor_to_plt)
from specvqgan.data.vggsound import CropImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = '2022-08-18T13-28-34_vas_transformer_nofeat'
log_dir = '/modelblob/users/v-chengw/models/specvqgan/'
config, sampler, melgan, melception = load_model(model_name, log_dir, device)

audio_fps = 22050
f = open('/home/yuwu1/samples/filelists/files.txt')
files = f.readlines()
f.close()
spec_path = '/home/yuwu1/samples/features/melspec_10s_22050hz'
output_path = '/home/yuwu1/samples/sampled/'
random_crop = False
crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)

for name in files:
    name = name.strip()
    spectrogram = np.load(f"{spec_path}/{name}_mel.npy")
    batch = {'input': spectrogram}
    batch = crop_img_fn(batch)
    feats = np.random.rand(1, 2048).astype(np.float32)
    batch['image'] =  2 * batch['input'] - 1
    batch['feature'] = feats
    batch = default_collate([batch])

    with torch.no_grad():
        ret = sampler.log_images(batch)
        samples = (ret['samples_half'] + 1)/2
        samples = melgan(samples[0])[0][0]
        sf.write(output_path+model_name+'-'+name+'_samplehalf.wav', samples.cpu().numpy(), 22050, 'FLOAT')



