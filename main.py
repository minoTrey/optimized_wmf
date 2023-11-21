import itertools
import time
import glob
import torch.utils.data
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

import models
from data_loader.KITTI_loader import KITTI2015

from util.img_generate import *
from util.utils import *


def wmf_optimize(args, inputs, outputs, loss_builder):
    loss = 0.0
    optim_params = []
    if args.save_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, inputs['name'][0].split('.')[0]))
    else:
        writer = None

    # Optimization target item is predicted disparity
    outputs['pred_disp'] = inputs['init_disp'].clone()
    optim_params.append((outputs['pred_disp']).requires_grad_(True))
    optimizer = optim.Adam(optim_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    with tqdm(range(args.iter), position=0, ncols=100) as pbar:
        for i in pbar:
            outputs['pred_disp'].grad = None

            # ToDO: WMF should be in build_loss
            loss = loss_builder.build_loss(args, inputs, outputs)
            print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            evals, outputs = get_epe_and_d1(args, inputs, outputs)
            # tensorboard
            if args.save_tensorboard:
                write_summary(args, writer, inputs, outputs, loss, evals, i)

    # print('\nEPE: {:04f} \t D1: {:04f} \t D1_ALL: {:04f}%'.format(loss['epe'], loss['d1'], loss['d_err']))

    return inputs, outputs, evals


def _main_(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Dataset Load
    if args.dataset == 'kitti':
        data_loader = torch.utils.data.DataLoader(KITTI2015(args),
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=16,
                                                  pin_memory=True,
                                                  drop_last=True)
    else:
        data_loader = None

    # For finding the best L1 loss weight
    # This is for additional loss weights in the future
    all_combinations = list(itertools.product(*(args.params[key] for key in args.params)))

    for idx, combination in enumerate(all_combinations):
        weights = weight_params(*combination)
        comb = 'l1 : {}\n'.format(weights['w_l1'])
        print('\n{}/{} combination\n{}'.format(idx + 1, len(all_combinations), comb))

        # parameters initialization
        epe_ori = []
        d1_ori = []
        d_err_ori = []
        epe = []
        d1 = []
        d_err = []
        n_samples = len(data_loader)

        data_time = AverageMeter()
        data_end = time.time()

        # optimize the input one by one
        for i, inputs in enumerate(data_loader):
            # path generation for logging
            args.config_dir = os.path.join('results/configs', args.test_name, 'experiment_{:03d}'.format(args.run_id))
            args.tensorboard_dir = os.path.join('results/tensorboard', args.test_name,
                                                'experiment_{:03d}'.format(args.run_id))
            args.save_dir = os.path.join('results/data', args.test_name, 'experiment_{:03d}'.format(args.run_id))
            os.makedirs(args.config_dir, exist_ok=True)
            os.makedirs(args.tensorboard_dir, exist_ok=True)
            os.makedirs(args.save_dir, exist_ok=True)

            args.w_l1 = weights['w_l1']

            # save configure arguments
            save_experiment_config(args)

            # parameters initialize for each data
            outputs = {}

            print('data name is {}'.format(inputs['left_name']))

            # loss builder
            loss_builder = getattr(models, args.loss_name)()
            data_time.update(time.time() - data_end)

            # Use CUDA
            if torch.cuda.is_available():
                for key, ipt in inputs.items():
                    if 'img' in key[0] or 'disp' in key[0] or 'gt' in key:
                        inputs[key] = ipt.cuda()
                    elif 'shape' in key:
                        inputs[key] = (inputs[key][0].item(), inputs[key][1].item())

            # Optimize
            inputs, outputs, evals = wmf_optimize(args, inputs, outputs, loss_builder)

            epe.append(evals['E'])
            epe_ori.append(evals['E_ori'])

            d1.append(evals['D1'])
            d1_ori.append(evals['D1_ori'])

            d_err.append(evals['D_err'])
            d_err_ori.append(evals['D_err_ori'])

            if args.save_image:
                eval_img_init = kitti_eval(args, D_gt=inputs['gt_disp_l'], D_est=inputs['init_disp'])
                topil(eval_img_init).save(os.path.join(args.save_dir, inputs['name'][0].split('.')[0]) + '_init.png')
                eval_img = kitti_eval(args, D_gt=inputs['gt'], D_est=outputs['pred_disp'])
                topil(eval_img).save(os.path.join(args.save_dir, inputs['name'][0].split('.')[0]) + '_est.png')

            save_excel(epe, d1, d_err, epe_ori, d1_ori, d_err_ori, args.save_dir)

        args.run_id += 1

        print(sum(epe) / n_samples)
        print(sum(d1) / n_samples)
        print(sum(d_err) / n_samples)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.test_name = 'wmf_optimize'

    args.device = 'cuda'
    args.iter = 50
    args.batch_size = 1
    args.learning_rate = 1e-4
    args.beta = 1
    args.print_freq = 1

    # options
    args.max_disp = 192
    args.height_ori = 375
    args.width_ori = 1242
    args.save_image = True
    args.save_video = False
    args.psmn_gen = False

    # path
    args.dataset = 'kitti'
    args.data_path = '/media/mino/SSD_8TB/DATASET/KITTI/4_Stereo/training'
    args.list_path = os.path.join(args.data_path, 'train_list_error_larger.txt')
    args.config_dir = 'results/configs'
    os.makedirs(os.path.join(args.config_dir, args.test_name), exist_ok=True)
    if len(os.listdir(os.path.join(args.config_dir, args.test_name))) == 0:
        args.run_id = 0
    else:
        runs = sorted(glob.glob(os.path.join(args.config_dir, args.test_name, 'experiment_*')))
        args.run_id = int(runs[-1].split('_')[-1]) + 1

    # use options
    args.use_l1 = True

    # Loss weight option
    args.params.update({'w_l1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})

    _main_(args)
