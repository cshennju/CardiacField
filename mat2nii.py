import nibabel as nib
import numpy as np
import scipy.io as sio
import os
import numpy as np

path = os.path.join('C:/Users/ASUS/XXX.mat')
mat = sio.loadmat(path) 
rgb = np.squeeze(mat['gray'])
new_image = nib.Nifti1Image(rgb,np.eye(4))
save_path = os.path.join('C:/Users/ASUS/XXX.nii.gz')
nib.save(new_image,save_path) 