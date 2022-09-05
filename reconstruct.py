import os
import soundfile as sf
import time
from pathlib import Path

import soundfile
import torch
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
model_name = '2022-08-17T05-30-11_audioset_codebook'
log_dir = '/modelblob/users/v-chengw/models/specvqgan/'
# loading the models might take a few minutes
config, sampler, melgan, melception = load_model(model_name, log_dir, device)

# Extract Spectrogram
audio_fps = 22050
video_paths=['/home/yuwu1/sample1/', '/home/yuwu1/sample2/', '/home/yuwu1/sample3/', '/home/yuwu1/sample4/']
for video_path in video_paths:
    spectrogram = extract_melspectrogram(video_path+'video.mp4', audio_fps)
    spectrogram = {'input': spectrogram}
    # [80, 860] -> [80, 848]
    random_crop = False
    crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
    spectrogram = crop_img_fn(spectrogram)

    # Prepare input
    batch = default_collate([spectrogram])
    batch['image'] = batch['input'].to(device)
    # Encode and Decode the Spectrogram
    with torch.no_grad():
        ret = sampler.log_images(batch)
        ret['inputs'] = (ret['inputs'] + 1) / 2
        ret['reconstructions'] = (ret['reconstructions'] + 1 ) / 2
        org = melgan(ret['inputs'][0])[0][0]
        rec = melgan(ret['reconstructions'][0])[0][0]
        sf.write(video_path+model_name+'_org.wav', org.cpu().numpy(), 22050, 'FLOAT')
        sf.write(video_path+model_name+'_rec.wav', rec.cpu().numpy(), 22050, 'FLOAT')



