import os
#from random import sample
#import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
#from pydub import AudioSegment
#from IPython.display import Audio, display
#from playsound import playsound
#from playsound import playsound
from torchvision import transforms
import psutil

# Dataset class
class audioDataset(Dataset):
    def __init__(self, classes, audio_dir, transform, num_samples, sample_rate, device):
        super().__init__()
        self.classes = classes
        self.audio_dir = audio_dir
        self.transform = transform
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.device = device

    def mem():
        process = psutil.Process() #initiate only once
        memory_info = process.memory_info()
        rss = memory_info.rss
        rss_mb = rss / (1024 * 1024)
        print(f"Memory usage: {rss_mb} MB")

    def __len__(self):
        return len(os.listdir(self.audio_dir))

    def __getitem__(self, index):
        # loading the audio files
        file_paths = os.listdir(self.audio_dir)
        audio_path = os.path.join(self.audio_dir, file_paths[index])
        signal, sr = torchaudio.load(audio_path)
        
        # preprocessing
        signal = self._mix_down_if_necessary(signal) 
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate, dtype=signal.dtype)
        signal = resampler(signal)
        signal = self.transform(signal)

        # add label
        label = self._add_label(file_paths[index])

        # transform, changes the input to 224x224 
        data_transforms = {
        "train": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}

        # changes the input to 3-channel
        trans = transforms.Lambda(lambda signal: signal.repeat(3, 1, 1) if signal.size(0)==1 else signal)
        signal = trans(signal)

        tr = data_transforms["train"]
        signal = tr(signal)
        #print("dataset")
        self.mem
        return signal, label

    # Preprocessing functions 

    def _cut_signal(self, signal): 
        cut_amount = self.num_samples / 4
        signal = signal[:, int(cut_amount):]
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            return signal
    
    # labels for gender and emotion classification    
    def _add_label(self, file):

        if self.classes == "emotion":
          if "_ANG_" in file or "_angry" in file:
              label = 0
          elif "_DIS_" in file or  "_disgust" in file:
              label = 1
          elif "_FEA_" in file or  "_fear" in file:
              label = 2
          elif "_HAP_" in file or  "_happy" in file:
              label = 3
          elif "_NEU_" in file or  "_neutral" in file:
              label = 4      
          elif "_SAD_" in file or  "_sad" in file:
              label = 5
          #else:
            #print("error emotion")
        if self.classes == "gender":
          if "FEMALE" in file:
            label = 0
          elif "MALE" in file:
            label = 1
          #else: 
            #print("error emotion")
        return 0

