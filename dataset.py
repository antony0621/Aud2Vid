import os
import torch
import numpy as np
import json
import cv2 as cv
import librosa
import torchaudio
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import DataLoader


# import matplotlib.pyplot as plt


def get_files(root, tag_json_path):  # obtain "class_name/tag.mp4/(mp3)" .mp4/ --> frame's dir; .mp3 --> audio's path
    tag_dict = json.load(open(tag_json_path, 'r'))  # {"acoustiic_guitar": [-6dwiJ-m4K4000001, -6dwiJ-m4K4000002...], ...}
    path_list = []
    for class_name in tag_dict.keys():
        for tag_name in tag_dict[class_name]:
            path_list.append(os.path.join(root, class_name, tag_name))

    return path_list


def get_frames_as_3d_tensor(frames_path):
    """

    :param frames_path: xxxxxxx.mp4/ contains 90 frames within it
    :return: Tensor (90, 3, 128, 128)
    """
    frame_list = []
    for frame_path in frames_path:
        frame = cv.imread(frame_path)  # 128 x 128 x 3
        frame = np.transpose(frame, (2, 0, 1))
        frame_list.append(frame)
    frame_3d_tensor = np.stack(frame_list, 0)
    frame_3d_tensor = frame_3d_tensor[30: 60]
    return frame_3d_tensor


def get_audio_as_tf_representation(audio_path, win_size=1022, hop_length=246, use_raw=False, use_mel=True):
    """
    Convert audio into temporal frequency representation by STFT with proper window size and hop length.
    :param use_raw: use raw audio signal as NN input, else transform into TF representation
    :param win_size:
    :param hop_length: 246 or 247 to ensure the result has 90 time steps, 3 time steps --> 1 frame
    :param audio_path: xxxxxxx.mp3
    :return: matrix: (1+win_size/2, length_audio/hop_length+1)
    """
    np_audio, sr = librosa.load(audio_path)  # how to avoid printing the f**king warning ?!
    if use_raw:
        return np_audio
    else:
        assert sr == 22050, "The sampling frequency is not required!"
        if len(np_audio) > 22050 * 3:
            # clip to 66150
            np_audio = np_audio[int((len(np_audio) - 66150)//2): int((len(np_audio) - 66150)//2) + 66150]
        elif len(np_audio) < 22050 * 3:
            # pad to 66150
            short = 66150 - len(np_audio)
            if short % 2 == 0:
                padding = np.zeros((short//2, ))
                np_audio = np.concatenate((padding, np_audio, padding), 0)
            else:
                padding1 = np.zeros((short//2, ))
                padding2 = np.zeros((short//2 + 1, ))
                np_audio = np.concatenate((padding1, np_audio, padding2), 0)
        np_audio = np_audio[22050: 22050 * 2]
        audio_tf_mat = librosa.stft(np_audio, n_fft=win_size, hop_length=hop_length)  # only use the magnitude ?
        mag, phase = np.abs(audio_tf_mat), np.angle(audio_tf_mat)
        if not use_mel:
            return audio_tf_mat, mag, phase
        else:
            # TODO: implement the Mel filter-bank code !
            pass


class MUSIC21(Dataset):
    def __init__(self, frame_path_list, audio_path_list):

        self.frame_path = frame_path_list
        self.audio_path = audio_path_list

        assert len(self.frame_path) == len(self.audio_path), "Inconsistency between audio and frames!"

    def __getitem__(self, index):
        audio_tf = get_audio_as_tf_representation(self.audio_path[index])  # audio temporal-frequent representation
        frames_3d = get_frames_as_3d_tensor(self.frame_path[index])  # N x 3 x 128 x 128
        audio_tf_tensor = torch.from_numpy(audio_tf)
        frames_3d_tensor = torch.from_numpy(frames_3d)
        return frames_3d_tensor, audio_tf_tensor

    def __len__(self):
        return len(self.frame_path)


def get_dataloader(root, tag_dir, validation_split=0.1, batch_size=256, is_training=True):
    frame_path_list, audio_path_list = get_files(root, tag_dir)
    dataset = MUSIC21(frame_path_list, audio_path_list)
    if is_training:
        size = len(dataset)
        raw_index = list(range(size))
        np.random.seed(1227)
        index = np.random.permutation(raw_index)
        split = int(np.floor((1 - validation_split) * size))

        train_index, val_index = index[:split], index[split:]

        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        _train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                   num_workers=8, pin_memory=True)
        _val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                 num_workers=8, pin_memory=True)

        return _train_loader, _val_loader, size
    else:
        test_sampler = SubsetRandomSampler(np.random.permutation(range(dataset.__len__())))
        _test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                  num_workers=8, pin_memory=True)

        return _test_loader, size


if __name__ == '__main__':
    train_tag_json = ""
    test_tag_json = ""
    training_set, validation_set = get_dataloader(root=os.getcwd(), tag_dir=train_tag_json, is_training=True)
    test_set = get_dataloader(root=os.getcwd(), tag_dir=test_tag_json, is_training=False)
