from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            img_idxs = np.random.choice(len(self.poses), self.batch_size)
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            gray = self.imgs[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'gray': gray[:, :1]}
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.imgs)>0: # if ground truth available
                gray = self.imgs[idx]
                sample['gray'] = gray[:, :1]
        return sample