import os
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import matplotlib as mpl
import pandas as pd
import torchvision.utils as vutils
from scipy.ndimage import grey_dilation
from matplotlib import cm

topil = transforms.ToPILImage()
totensor = transforms.ToTensor()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def get_args_parser():
    parser = argparse.ArgumentParser()

    # System
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    # Train
    parser.add_argument('--loss_name', default='WMF_Loss', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--iter', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)

    # Options
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--height', default=375, type=int)
    parser.add_argument('--width', default=1242, type=int)
    parser.add_argument('--test_name', default='WMF_optimize', type=str)

    # Toggle
    parser.add_argument('--save_image', default=False, type=bool)
    parser.add_argument('--save_video', default=False, type=bool)
    parser.add_argument('--use_l1', default=True, type=bool)

    # HyperParams
    parser.add_argument('--params', default={'w_l1': [1]})
    parser.add_argument('--beta', default=0.01, type=float)

    # Path
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--list_path', default='', type=str)
    parser.add_argument('--checkpoint_path', default='', type=str)
    parser.add_argument('--save_dir', default='', type=str)
    parser.add_argument('--config_dir', default='', type=str)
    parser.add_argument('--tensorboard_dir', default='', type=str)

    return parser


def get_epe_and_d1(args, inputs, outputs):
    def get_evaluation(gt, est):
        mask = (gt * 256 > 0) & (gt * 256 < args.max_disp)
        Est = torch.zeros_like(est)
        Est[mask] = est[mask]
        gt *= 256
        Est *= 255
        E = torch.abs(gt - Est)
        mask_err = (E > 3) & (E / gt > 0.05)
        D1 = torch.mean(mask_err.float())
        n_err = torch.sum((gt > 0) & (E > 3) & ((E / gt) > 0.05))
        n_total = torch.sum(gt > 0)
        D_err = n_err / n_total
        E_map = torch.zeros_like(E)
        E_map[mask_err] = E[mask_err]
        # print('\n[{}]:\t EPE: {:04f} \t D1: {:04f} \t D1_: {:04f}%'.format(mode, E.mean(), D1, D_err * 100))
        return E.mean(), D1, D_err * 100, E_map

    evals = {}

    d_gt = inputs['gt'].clone()
    d_est_ori = inputs['init_disp'].clone()
    d_est = outputs['pred_disp']
    evals['E_ori'], evals['D1_ori'], evals['D_err_ori'], outputs['E_map_ori'] = get_evaluation(d_gt, d_est_ori)
    evals['E'], evals['D1'], evals['D_err'], outputs['E_map'] = get_evaluation(d_gt, d_est)

    return evals, outputs


def error_colormap():
    cols = np.array([
        [0/3.0,       0.1875/3.0,  49,  54, 149],
        [0.1875/3.0,  0.375/3.0,   69, 117, 180],
        [0.375/3.0,   0.75/3.0,   116, 173, 209],
        [0.75/3.0,    1.5/3.0,    171, 217, 233],
        [1.5/3.0,     3/3.0,      224, 243, 248],
        [3/3.0,       6/3.0,      254, 224, 144],
        [6/3.0,      12/3.0,      253, 174,  97],
        [12/3.0,      24/3.0,      244, 109,  67],
        [24/3.0,      48/3.0,      215,  48,  39],
        [48/3.0,     np.inf,      165,   0,  38 ]
    ])
    cols[:, 2:5] /= 255
    return cols


def disp_error(D_gt, D_est, tau):
    E = np.abs(D_gt - D_est)
    D_gt_mask = D_gt > 0
    E_tau_1 = np.zeros_like(E)
    E_tau_1[D_gt_mask] = E[D_gt_mask] / D_gt[D_gt_mask]
    E_mask = (E > tau[0]) & (E_tau_1 > tau[1])
    n_err = np.sum(D_gt_mask & E_mask)
    n_total = np.sum(D_gt_mask)
    d_err = n_err / n_total
    return d_err


def disp_error_map(D_gt, D_est):
    D_gt_val = D_gt > 0
    E = np.abs(D_gt - D_est)
    E[~D_gt_val] = 0
    return E, D_gt_val


def disp_error_image(D_gt, D_est, tau, dilate_radius=1):
    E, D_val = disp_error_map(D_gt, D_est)
    mask = D_gt > 0
    E_tau_1 = np.zeros_like(E)
    E_tau_1[mask] = E[mask] / D_gt[mask] / tau[1]
    E = np.minimum(E / tau[0], E_tau_1)
    cols = error_colormap()
    D_err = np.zeros(list(D_gt.shape) + [3])
    for i in range(cols.shape[0]):
        mask = (D_val > 0) & (E >= cols[i, 0]) & (E <= cols[i, 1])
        for j in range(3):
            D_err[mask, j] = cols[i, j+2]
    D_err = grey_dilation(D_err, dilate_radius)
    return D_err.squeeze(1)


def disp_to_color(disp, max_disp=None):
    (height, width) = disp.shape
    zero_mask = disp == 0
    disp[zero_mask] = disp[zero_mask] - 1
    if max_disp is None:
        max_disp = np.max(disp)
    disp_norm = disp / max_disp
    disp_min = np.minimum(disp_norm, 1)
    disp_flat = disp_min.flatten()

    map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]], dtype=np.float32)
    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]

    I_repeated = np.tile(disp_flat.reshape(1, -1), (6, 1))
    cbins_repeated = np.tile(cbins[:, np.newaxis], (1, np.prod(I_repeated.shape[1])))
    comparison_matrix = (I_repeated > cbins_repeated).astype(np.uint)
    sum_comparison = np.sum(comparison_matrix, axis=0)
    limited_sum_comparison = np.minimum(sum_comparison, 6)
    ind = limited_sum_comparison

    bins = 1. / bins
    cbins = np.concatenate(([0], cbins))
    for i, v in enumerate(ind):
        disp_flat[i] = (disp_flat[i] - cbins[v]) * bins[v]

    # index mapping
    mapping = map[ind, :3]
    mapping_ = map[ind + 1, :3]
    # flip disp value
    disp_flip = np.tile((1 - disp_flat)[:, np.newaxis], (1, 3))
    disp_not_flip = np.tile(disp_flat[:, np.newaxis], (1, 3))

    # 3-maximum value extract
    disp_sum = (mapping * disp_flip) + (mapping_ * disp_not_flip)
    disp_max = np.maximum(disp_sum, 0)
    disp_min_ = np.minimum(disp_max, 1)

    color_disp = disp_min_.reshape(height, width, 3)
    return (color_disp * 255).astype(np.uint8)


def kitti_eval(args, D_gt, D_est):
    # error threshold
    tau = [3, 0.05]

    # Convert numpy and 0~1 to 0~256 range
    D_gt = D_gt.detach().cpu().numpy() * 255
    D_est = D_est.detach().cpu().numpy() * 255

    # error value check
    d_err = disp_error(D_gt, D_est, tau)

    # error image generation as followed KITTI SDK
    D_err = disp_error_image(D_gt, D_est, tau)
    D_err = (D_err[0]*255).astype(np.uint8)

    # D_err_img = topil((D_err[0]*256).astype(np.uint8))
    # D_err_img.show()

    # Prepare image for display
    D_color = np.concatenate((D_est.squeeze(), D_gt.squeeze()))
    D_color = disp_to_color(D_color, max_disp=args.max_disp)

    display_img = np.concatenate([D_color, D_err], axis=0)

    # topil(display_img).show()
    # print(f'Disparity Error: {d_err * 100:.2f} %')
    return display_img


def weight_params(w_l1):
    weights = {'w_l1': w_l1}
    return weights


def depth2color(img):
    img = img.detach().cpu().numpy().squeeze()
    vmax = np.percentile(img, 95)
    normalizer = mpl.colors.Normalize(vmin=img.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    colormapped_im = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    # colormapped_im = Image.fromarray(colormapped_im)
    colormapped_im = totensor(colormapped_im)
    return colormapped_im


def write_summary(args, writer, inputs, outputs, losses, evals, step):
    """Write an event to the tensorboard events file
    """

    # Images
    # input images
    colors = [denormalize(inputs['img_l'][0]), denormalize(inputs['img_r'][0])]
    colors = vutils.make_grid(colors, padding=0, nrow=1, scale_each=True)
    writer.add_image("colors", colors, step)

    # gt disparity
    # original lidar
    gt = [inputs['gt'][0]]
    gt = vutils.make_grid(gt, padding=0, nrow=1, scale_each=True)
    writer.add_image("gt", gt, step)

    # initial disparity
    disp_init = [inputs['init_disp'][0]]
    disp_init = vutils.make_grid(disp_init, padding=0, nrow=1, scale_each=True)
    writer.add_image("disp_init", disp_init, step)

    # pred disp images
    disp_est = [outputs['disp_est_l'][0]]
    disp_est = vutils.make_grid(disp_est, padding=0, nrow=1, scale_each=True)
    writer.add_image("disp_est", disp_est, step)

    # error image
    error_map = [outputs['E_map'][0], outputs['E_map_ori'][0]]
    error_map = vutils.make_grid(error_map, padding=0, nrow=1, scale_each=True)
    writer.add_image("error_map", error_map, step)

    # Scalar
    for l, v in losses.items():
        if isinstance(l, tuple):
            writer.add_scalar("loss/{}".format(l[0]), v, step)
        else:
            writer.add_scalar("loss/{}".format(l), v, step)

    for k, v in evals.items():
        if isinstance(k, tuple):
            writer.add_scalar("error/{}".format(k[0]), v, step)
        else:
            writer.add_scalar("error/{}".format(k), v, step)


def save_excel(epe, d1, d_err,
               epe_ori, d1_ori, d_err_ori,
               path):

    def extract_np(evals):
        e_np = [e.detach().cpu().numpy() for e in evals]
        e_np = np.array(e_np)
        return e_np

    epe = extract_np(epe)
    epe_ori = extract_np(epe_ori)

    d1 = extract_np(d1)
    d1_ori = extract_np(d1_ori)

    d_err = extract_np(d_err)
    d_err_ori = extract_np(d_err_ori)

    filename = os.path.join(path, 'eval.xlsx')
    df = pd.DataFrame({'EPE': epe, 'D1': d1, 'D_ERR': d_err,
                       'EPE_ORI': epe_ori, 'D1_ORI': d1_ori, 'D_ERR_ORI': d_err_ori})
    df.to_excel(filename, index=False)


def save_experiment_config(args):
    with open(os.path.join(args.config_dir, 'config.cfg'), 'w') as file:
        config_dict = vars(args)
        for k in vars(args):
            file.write(f"{k}={config_dict[k]} \n")


# def save_video(args, images, losses):
#     def create_graph(i):
#         plt.clf()
#
#         EPOCH = len(disps)
#
#         # Image axis
#         ax_gt = plt.subplot(5, 3, 1)
#         ax_gt.imshow(ToPILImage()(gt_disp_l), cmap='gray')
#         ax_gt.set_title("Ground Truth")
#
#         ax_ori = plt.subplot(5, 3, 2)
#         ax_ori.imshow(img_l)
#         ax_ori.set_title("Original Image")
#
#         ax_init = plt.subplot(5, 3, 3)
#         ax_init.imshow(pred_disp, cmap='gray')
#         ax_init.set_title("Initial Disparity")
#
#         ax_ori = plt.subplot(5, 3, 4)
#         ax_ori.imshow(ToPILImage()(left_img_w[i]))
#         ax_ori.set_title("Left warped")
#
#         ax_ori = plt.subplot(5, 3, 5)
#         ax_ori.imshow(ToPILImage()(right_img_w[i]))
#         ax_ori.set_title("Right warped")
#
#         ax_est = plt.subplot(5, 3, 6)
#         ax_est.imshow(disps[i], cmap='gray')
#         ax_est.set_title("Refined Disparity")
#
#         # Loss axis
#         ax1 = plt.subplot(5, 1, 3)  # row, column, index
#         ax2 = ax1.twinx()
#
#         ax3 = plt.subplot(5, 1, 4)
#         ax4 = ax3.twinx()
#
#         ax5 = plt.subplot(5, 1, 5)
#         ax6 = ax5.twinx()
#
#         # L1 loss
#         p1 = ax1.plot(range(1, EPOCH + 1)[:i + 1], l1_history[:i + 1], 'r')  # red
#         # Smooth
#         p2 = ax2.plot(range(1, EPOCH + 1)[:i + 1], smooth_history[:i + 1], 'b')  # blue
#
#         # disp Smoothness
#         p3 = ax3.plot(range(1, EPOCH + 1)[:i + 1], disp_smooth_history[:i + 1], 'c')  # cyan
#         # Photometric
#         p4 = ax4.plot(range(1, EPOCH + 1)[:i + 1], photometric_history[:i + 1], 'r')  # red
#
#         # Evaluations
#         p5 = ax5.plot(range(1, EPOCH + 1)[:i + 1], epe_history[:i + 1], 'y')  # yellow
#         p6 = ax6.plot(range(1, EPOCH + 1)[:i + 1], d1_history[:i + 1], 'k')  # black
#
#         ax1.set_xlim([-5, EPOCH + 5])
#         ax3.set_xlim([-5, EPOCH + 5])
#         ax5.set_xlim([-5, EPOCH + 5])
#         # ax1.set_ylim([min(l1_history) * 0.7, max(loss_history) * 1.1])
#         # ax2.set_ylim([0, 100])
#
#         ax1.set_xlabel("Epochs")
#         ax1.set_ylabel("L1")
#         ax2.set_ylabel("Smoothness")
#
#         ax3.set_xlabel("Epochs")
#         ax3.set_ylabel("regularization")
#         ax4.set_ylabel("Photometric")
#
#         ax5.set_xlabel("Epochs")
#         ax5.set_ylabel("EPE")
#         ax6.set_ylabel("D1")
#
#         ax1.set_title(
#             "L1: {:04f}      Smoothness: {:04f}".format(torch.mean(l1_history[i]), torch.mean(smooth_history[i])))
#         ax1.grid()
#         ax1.legend(p1 + p2, ["L1", "Smoothness"], loc="right")
#
#         ax3.set_title("regularization: {:04f}      Photometric: {:04f}".format(torch.mean(disp_smooth_history[i]),
#                                                                             torch.mean(photometric_history[i])))
#         ax3.grid()
#         ax3.legend(p3 + p4, ["regularization", "Photometric"], loc="right")
#
#         ax5.set_title("EPE: {:04f}      D1: {:04f}".format(torch.mean(epe_history[i]), torch.mean(d1_history[i])))
#         ax5.grid()
#         ax5.legend(p5 + p6, ["EPE", "D1"], loc="right")
#
#         plt.tight_layout()
#
#     from matplotlib.animation import FuncAnimation
#
#     os.makedirs('results/video', exist_ok=True)
#
#     fig = plt.figure(figsize=[16, 9])
#     plt.suptitle("Conv(1,3)")
#
#     ani = FuncAnimation(fig=fig, func=create_graph, frames=args.iter, interval=50)
#     # plt.show()
#     ani.save(args.save_dir, fps=30)
