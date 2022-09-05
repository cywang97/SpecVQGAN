import os
import time
import pdb
import numpy as np
import soundfile as sf

from pathlib import Path

import IPython.display as display_audio
import soundfile
import torch
import pickle
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
from specvqgan.data.vggsound import CropImage, CropFeats
from train import instantiate_from_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = '2022-08-19T02-50-18_vas_transformer'
log_dir = '/modelblob/users/v-chengw/models/specvqgan/'
config, sampler, melgan, melception = load_model(model_name, log_dir, device)

audio_fps = 22050
f = open('/home/yuwu1/samples/filelists/files.txt')
files = f.readlines()
f.close()
spec_path = '/home/yuwu1/samples/features/melspec_10s_22050hz'
rgb_path = '/home/yuwu1/samples/features/feature_rgb_bninception_dim1024_21.5fps'
flow_path = '/home/yuwu1/samples/features/feature_flow_bninception_dim1024_21.5fps' 
output_path = '/home/yuwu1/samples/sampled/'
random_crop = False
crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
crop_feat_fn = CropFeats([config.data.params.feat_crop_len, config.data.params.feat_depth], random_crop)
feat_sampler = instantiate_from_config(config.data.params.feat_sampler_cfg)

for name in files:
    name = name.strip()
    spectrogram = np.load(f"{spec_path}/{name}_mel.npy")
    batch = {'input': spectrogram}
    batch = crop_img_fn(batch)
    rgb_feat = pickle.load(open(f"{rgb_path}/{name}.pkl", 'rb'), encoding='bytes')
    flow_feat = pickle.load(open(f"{flow_path}/{name}.pkl",'rb'), encoding='bytes')
    feats = np.concatenate((rgb_feat, flow_feat), axis=1)
    feats_padded = np.zeros((config.data.params.feat_len, feats.shape[1]))
    feats_padded[:feats.shape[0], :] = feats[:config.data.params.feat_len, :]
    batch['image'] =  2 * batch['input'] - 1
    batch['feature'] = feats
    batch = crop_feat_fn(batch)
    batch = feat_sampler(batch)
    batch = default_collate([batch])

    with torch.no_grad():
        ret = sampler.log_images(batch)
        samples = (ret['samples_half'] + 1)/2
        samples = melgan(samples[0])[0][0]
        sf.write(output_path+model_name+'-'+name+'_samplehalf.wav', samples.cpu().numpy(), 22050, 'FLOAT')
        samples = (ret['samples_nopix'] + 1)/2
        samples = melgan(samples[0])[0][0]
        sf.write(output_path+model_name+'-'+name+'_samplenopix.wav', samples.cpu().numpy(), 22050, 'FLOAT')
        samples = (ret['samples_det'] + 1)/2
        samples = melgan(samples[0])[0][0]
        sf.write(output_path+model_name+'-'+name+'_sampledet.wav', samples.cpu().numpy(), 22050, 'FLOAT')
        samples = (ret['reconstructions'] + 1)/2
        samples = melgan(samples[0])[0][0]
        sf.write(output_path+model_name+'-'+name+'_rec.wav', samples.cpu().numpy(), 22050, 'FLOAT')




