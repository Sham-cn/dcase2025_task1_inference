import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

from models.cp_mobile import get_cpmobile
from dataset.dcase25 import get_tau25_evalset
from models.helpers.complexity import get_torch_macs_memory

index_to_label = {0: 'airport', 1: 'bus', 2: 'metro', 3: 'metro_station', 4: 'park', 5: 'public_square', 6: 'shopping_mall', 7: 'street_pedestrian', 8: 'street_traffic', 9: 'tram'}

class MelForward(nn.Module):
    def __init__(self, config, is_train=False):
        super(MelForward, self).__init__()
        self.sample_rate = config.sample_rate
        self.is_train = is_train
        self.constant_freq = config.constant_freq

        self.resample = torchaudio.transforms.Resample(
            orig_freq=config.orig_sample_rate,
            new_freq=config.sample_rate
        )

        self.mel_process = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        self.freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        self.timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

    def forward(self, x, sample_rate):
        if self.constant_freq:
            # x = x.reshape(-1)
            # x = x.contiguous()
            x = self.resample(x)
            x = self.mel_process(x)
            if self.is_train:
                x = self.freqm(x)
                x = self.timem(x)
            output = (x + 1e-5).log()
        else:
            len_x = x.shape[0]
            outputs = []
            for i in range(len_x):
                single_x = x[i].unsqueeze(0)
                single_sample_rate = sample_rate[i].item()
                resampled_x = torchaudio.functional.resample(
                    single_x,
                    orig_freq=single_sample_rate,
                    new_freq=self.sample_rate
                )
                mel_x = self.mel_process(resampled_x)
                if self.is_train:
                    mel_x = self.freqm(mel_x)
                    mel_x = self.timem(mel_x)
                log_mel_x = (mel_x + 1e-5).log()
                outputs.append(log_mel_x)
                output = torch.cat(outputs, dim=0)
        return output

def inference(config):
    model = get_cpmobile(n_classes=config.class_size).to(config.device)
    ckpt_dir = config.ckpt_dir
    ckpt_id = config.ckpt_id
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_id))
    model.load_state_dict(ckpt)
    output_path = f"{config.teacher}_output.csv"

    eval_set = get_tau25_evalset()
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

    mel = MelForward(config, is_train=False).to(config.device)

    # Compute model complexity (MACs, parameters) and log to W&B
    sample = next(iter(eval_loader))[0][0].unsqueeze(0).to(config.device)  # Single sample
    shape = mel(sample, 44100).size()
    macs, params_bytes = get_torch_macs_memory(model, input_size=shape)
    print(f'macs: {macs}, params: {params_bytes}')

    files_name = []
    logits = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(eval_loader):
            inputs = data[0].to(config.device)
            samplerate = data[1].to(config.device)
            files = data[2]
            devices = data[3]

            inputs = mel(inputs, samplerate)
            outputs = model(inputs)

            files_name.append(files)
            logits.append(outputs)

    logits = torch.cat([batch_logits.cpu() for batch_logits in logits], dim=0)
    filenames = [filename for batch_filenames in files_name for filename in batch_filenames]

    assert logits.shape[1] == config.class_size, f"类别数不匹配: {logits.shape[1]} != {config.class_size}"

    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = [index_to_label[idx] for idx in predicted_indices]

    data = []
    for i, filename in enumerate(filenames):
        row = [filename, predicted_labels[i]]  # 文件名和预测标签
        row.extend(probabilities[i].tolist())  # 每个类别的概率
        data.append(row)

    columns = ['filename', 'scene_label'] + list(index_to_label.values())

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, sep='\t', na_rep='nan', float_format='%.4f', index=False)

    print(f"CSV文件已保存至: {output_path}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE-2025-Task1')
    # basic parameters
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="Checkpoint Dir for loading base model")
    parser.add_argument("--ckpt_id", type=str, default="cnn/net.pkl", help="Checkpoint ID for loading base model")
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--teacher', type=str, default='cnn', choices=['cnn', 'transformer', 'all', 'all_v2'])
    parser.add_argument('--seed', type=int, default=42)

    # parameters for Mel Spectrogram
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)
    parser.add_argument('--hop_length', type=int, default=500)
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--n_mels', type=int, default=256)
    parser.add_argument('--freqm', type=int, default=48)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=None)

    # parameters for dataset
    parser.add_argument('--class_size', type=int, default=10)
    parser.add_argument('--constant_freq', action='store_false', help='Default True')
    parser.add_argument('--orig_sample_rate', type=int, default=44100)

    args = parser.parse_args()

    inference(args)
