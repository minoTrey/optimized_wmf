import os
import numpy as np


datapath = os.path.join('/media/mino/SSD_8TB/DATASET/KITTI/4_Stereo/training')
image_path = os.path.join(datapath, 'image_2')
image_list = [filename for filename in sorted(os.listdir(image_path))]
np.savetxt(os.path.join(datapath, 'train_list.txt'), np.array(image_list), fmt='%s')
