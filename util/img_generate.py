import os
import cv2
import models
import torch.utils.data
from data_loader.KITTI_loader import KITTI2015
from util.utils import *
from PIL import Image
from torchvision.transforms import ToPILImage as topil
import torch.nn.functional as F
import torch.nn as nn


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.psmn_gen = True

    args.batch_size = 1

    # Stereo matching methods
    args.method = "PSMNet"  # "SGBM"
    args.pretrained = True

    args.max_disp = 192
    args.width = 1024
    args.height = 320
    assert args.width % 32 == 0, "'width' must be a multiple of 32"
    assert args.height % 32 == 0, "'height' must be a multiple of 32"

    # Path
    args.data_path = '/media/mino/SSD_8TB/DATASET/KITTI/4_Stereo/training'
    args.list_path = os.path.join(args.data_path, 'train_list_all.txt')
    args.save_path = os.path.join(args.data_path, args.method)
    os.makedirs(args.save_path, exist_ok=True)
    pretrained_model = 'pretrained_model_KITTI2015.tar'
    args.pretrained_path = os.path.join('../pretrained', pretrained_model)

    data_loader = torch.utils.data.DataLoader(KITTI2015(args),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=16,
                                              pin_memory=True,
                                              drop_last=True)

    # Model load
    if args.method == 'PSMNet':
        model = getattr(models, args.method)(args.max_disp)
        # it has to use dataparallel if you want to use pretrained model. load_state_dict(strict=False) is not working
        model = nn.DataParallel(model, device_ids=[0])
    else:
        model = getattr(models, args.method)()

    if torch.cuda.is_available():
        model = model.to(device='cuda')
    else:
        model = model.to(device='cpu')

    if args.pretrained:
        pretrained_dict = torch.load(args.pretrained_path)
        # model.load_state_dict(pretrained_dict['state_dict'], strict=False)
        model.load_state_dict(pretrained_dict['state_dict'])

    model.eval()

    for i, inputs in enumerate(data_loader):
        if torch.cuda.is_available():
            for key, ipt in inputs.items():
                if 'img' in key or 'disp' in key or 'gt' in key or 'init_disp' in key:
                    inputs[key] = ipt.cuda()
                elif 'pad' in key:
                    inputs[key] = int(ipt)

        pred_disp = model(inputs)
        if args.method == 'PSMNet':
            if inputs['top_pad'] != 0 and inputs['right_pad'] != 0:
                pred_disp = pred_disp[:, :, inputs['top_pad']:, :-inputs['right_pad']]
            elif inputs['top_pad'] == 0 and inputs['right_pad'] != 0:
                pred_disp = pred_disp[:, :, :, :-inputs['right_pad']]
            elif inputs['top_pad'] != 0 and inputs['right_pad'] == 0:
                pred_disp = pred_disp[:, :, inputs['top_pad']:, :]

        disp_save = pred_disp.squeeze().detach().cpu().numpy() / args.disp_max
        # disp_save = pred_disp.squeeze().detach().cpu().numpy()
        disp_save = disp_save * 255 * 256
        disp_save = disp_save.astype(np.uint16)
        disp_save = Image.fromarray(disp_save)
        disp_save.save(os.path.join(args.save_path, inputs['name'][0]))
