from .heart import HeartDataset
from .heart_dy import HeartDYDataset
from .heart_dy_3d import HeartDY3dDataset


dataset_dict = {'heart': HeartDataset,
                'heart_dy':HeartDYDataset,
                'heart_dy_3d':HeartDY3dDataset}