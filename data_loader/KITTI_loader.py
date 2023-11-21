from torch.utils.data import Dataset
import os
from PIL import Image
from . import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KITTI2015(Dataset):
    def __init__(self, args):
        self.left_img_path = {}
        self.right_img_path = {}
        self.init_disp_path = {}
        self.gt_path = {}
        self.args = args
        if args.psmn_gen:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        with open(args.list_path, "r") as f:
            self.list = sorted(f.readlines())
        for i, line in enumerate(self.list):
            line = line.strip()
            self.left_img_path[i] = os.path.join(args.data_path, 'image_2', line)
            self.right_img_path[i] = os.path.join(args.data_path, 'image_3', line)
            if not args.psmn_gen:
                self.init_disp_path[i] = os.path.join(args.data_path, 'PSMNet', line)
                self.gt_path[i] = os.path.join(args.data_path, 'disp_occ_0', line)
            # self.calib_path = os.path.join(self.data_path, 'data_scene_flow_calib/training/calib_cam_to_cam',
            #                                line.replace('_10', '').replace('png', 'txt'))

    def read_img(self, path):
        try:
            img = Image.open(path)
            return img
        except Exception as e:
            print(e)
            print('{} is not exist'.format(path))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        inputs = {}

        inputs['img_l'] = self.read_img(self.left_img_path[idx])
        inputs['img_r'] = self.read_img(self.right_img_path[idx])
        if not self.args.psmn_gen:
            inputs['init_disp'] = self.read_img(self.init_disp_path[idx])
            inputs['gt'] = self.read_img(self.gt_path[idx])

        inputs['name'] = os.path.basename(self.left_img_path[idx])

        inputs = self.transform(inputs)
        inputs['shape_ori'] = (self.read_img(self.left_img_path[idx])).size[::-1]

        # Padding
        if self.args.psmn_gen:
            # padding
            if inputs['img_l'].shape[1] % 16 != 0:
                times = inputs['img_l'].shape[1] // 16
                inputs['top_pad'] = (times + 1) * 16 - inputs['img_l'].shape[1]
            else:
                inputs['top_pad'] = 0

            if inputs['img_l'].shape[2] % 16 != 0:
                times = inputs['img_l'].shape[2] // 16
                inputs['right_pad'] = (times + 1) * 16 - inputs['img_l'].shape[2]
            else:
                inputs['right_pad'] = 0
            inputs = transforms.Pad(inputs['top_pad'], inputs['right_pad'])(inputs)

        return inputs
