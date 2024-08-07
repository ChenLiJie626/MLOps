import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import nibabel as nib
import myvi
import numpy as np
import scipy.ndimage as ndimg

# 使用原始字符串表示路径
file_path = r'D:\DeepLearning\show\00_pred.nii.gz'
print(file_path)  # 打印路径以确认

pnii = nib.load(file_path)
pimgs = pnii.get_fdata()

pzoom = pnii.header.get_zooms()

# 调用 build_surf3d 函数
pvts, pfs, pns, pvs = myvi.util.build_surf3d(pimgs, ds=1, level=0.5, spacing=pzoom)

manager = myvi.Manager()
manager.add_surf('pspleen', pvts, pfs, pns, (1, 0, 0))

manager.show('Organ 3D Demo')
