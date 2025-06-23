import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List

# dataset_dir = '/SATA02/chenxy/DataSet/AudioSet'
dataset_dir = '/SATA04/environmental_classification/AudioSet'
assert dataset_dir, "Specify 'AudioSet' dataset location in 'dataset_dir'."

dataset_config = {
    "dataset_name": "audioset",
    "train_segment_path": 'audios/balanced_train_segments',
    "eval_segment_path": 'audios/eval_segments',
    "unbalanced_segment_path": 'audios/unbalanced_train_segments',
    "meta_path": "metadata",
    "label_csv": "class_labels_indices.csv",
    "train_csv": "balanced_train_segments.csv",
    "eval_csv": "eval_segments.csv",
    "num_class": 527,
    "sample_rate": 32000,
    "audio_length": 10,
}

# get labels
label_csv_path = os.path.join(dataset_dir, dataset_config["meta_path"], dataset_config["label_csv"])
label_df = pd.read_csv(label_csv_path)
print(label_df.keys())
labels = label_df["mid"].to_list()

# def label_to_index(words: list):
#     return [torch.tensor(labels.index(word)) for word in words]

def index_to_label(index):
    return labels[index]

def label_to_index(words: list):
    num_class = len(labels)
    label = torch.zeros(num_class, dtype=torch.float32)
    for word in words:
        label[labels.index(word)] = 1.0
    return label

class AudioSetDataset(Dataset):
    def __init__(self, meta_csv: str, subset: str):
        self.subset = subset
        names = ['YTID', 'start', 'end', 'labels']
        df = pd.read_csv(meta_csv, sep=', ', skiprows=3, header=None, names=names, engine='python')
        self.files = []
        # self.start_time = df['start'].values
        # self.end_time = df['end'].values
        self.labels = []
        self.absent_file = []
        self.target_length = 320000
        for idx, row in df.iterrows():
            file_name = row['YTID']
            labels = list(row['labels'].replace('"', '').split(','))
            audio_path = os.path.join(dataset_dir, 'audios', self.subset, f'Y{file_name}.wav')
            if os.path.exists(audio_path):
                self.files.append(file_name)
                self.labels.append(labels)
            else:
                self.absent_file.append(file_name)

    def __getitem__(self, index: int):
        audio_path = os.path.join(dataset_dir, 'audios', self.subset, f'Y{self.files[index]}.wav')
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(1) > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            pad_len = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        return waveform, sample_rate, self.files[index], label_to_index(self.labels[index])

    def __len__(self) -> int:
        return len(self.files)

class TimeShiftDataset(Dataset):
    """
    A dataset implementing time shifting of waveforms.
    """

    def __init__(self, dataset: Dataset, shift_range: int, axis: int = 1):
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index: int):
        waveform, sample_rate, file, label = self.dataset[index]
        shift = np.random.randint(-self.shift_range, self.shift_range + 1)
        return waveform.roll(shift, self.axis), sample_rate, file, label

    def __len__(self) -> int:
        return len(self.dataset)

def get_dataset(meta_csv, subset, roll: int = 0) -> Dataset:
    """
    Returns the training dataset for a specified data split percentage.

    Args:
        subset: The subset chosen in ["balanced_train_segments", "unbalanced_train_segments", "eval_segments"]
        roll (int): Time shift range.
    """
    dataset = AudioSetDataset(meta_csv, subset)
    return TimeShiftDataset(dataset, shift_range=roll) if roll else dataset

def get_audioset_dataset(roll: int = 0):
    train_csv = os.path.join(dataset_dir, dataset_config["meta_path"], dataset_config["train_csv"])
    eval_csv = os.path.join(dataset_dir, dataset_config["meta_path"], dataset_config["eval_csv"])
    train_dataset = get_dataset(train_csv, subset="balanced_train_segments", roll=roll)
    eval_dataset = get_dataset(eval_csv, subset="eval_segments", roll=roll)
    test_dataset = get_dataset(eval_csv, subset="eval_segments", roll=roll)

    return train_dataset, eval_dataset, test_dataset









if __name__ == "__main__":
    flag = 3
    if flag == 1:
        pass

    if flag == 2:
        train_set, eval_set, test_set = get_audioset_dataset()
        print(len(train_set), len(eval_set), len(test_set))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    if flag == 3:
        from tqdm import tqdm
        train_set, eval_set, test_set = get_audioset_dataset()
        abn = []
        # loader3 = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        # for step, data in tqdm(enumerate(loader3)):
        #     if data[1] != 320000:
        #         abn.append((data[2], data[1]))

        wf = set()
        sr = set()
        # for i in tqdm(range(len(train_set))):
        for i in range(5):
            waveform, sample_rate, _, label = train_set[i]
            # wf.add(waveform.shape)
            # sr.add(sample_rate)
            # print(wf)
            print(label)
        print(wf)
        print(sr)








