import torch
from torch.utils.data import Dataset

import numpy as np
import os
import os.path as osp
from PIL import Image

from torchvision import transforms as T

class TryonDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.define_transforms()
        
        self.image = []
        self.agnostic = []
        self.person_pose = []
        self.garment_pose = []
        self.garmented = []
        
        for i in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir ,i)
            
        
    def read_meta(self):
        pass
        
    def define_transforms(self):
        self.transform = T.Compose([
                T.ToTensor()
            ])
        
    def __len__(self):
        return len(os.listdir(self.root_dir))

    
    def __getitem__(self,idx):

        image = []
        agnostic = []
        seg = []
        p_pose = []
        s_pose = []

        
        for i in os.listdir(self.root_dir):

            path = osp.join(self.root_dir, i)
            image.append(self.transform(np.array(Image.open(osp.join(path,'image.png')))))
            agnostic.append(self.transform(np.array(Image.open(osp.join(path,'agnostic.png')))))
            seg.append(self.transform(np.array(Image.open(osp.join(path,'seg.png')))))
            p_pose.append(self.transform(np.array(Image.open(osp.join(path,'personpose.png')))))
            s_pose.append(self.transform(np.array(Image.open(osp.join(path,'segmentpose.png')))))

        sample = {
            'image': image[idx],
            'agnostic': agnostic[idx],
            'segment': seg[idx],
            'personpose': p_pose[idx],
            'garmentpose': s_pose[idx],
        }

        return sample
      