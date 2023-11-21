'''
3D Visualization code sample
'''

import open3d as o3d
from util.utils import *


def load_path(paths, n_pod):
    test_list = np.loadtxt(args.list_path, dtype=str)
    for pod_idx in range(n_pod):
        paths['left_img_path', pod_idx] = [os.path.join(args.left_img_path[pod_idx], i) for i in test_list]
        paths['lidar_gt_path', pod_idx] = [os.path.join(args.lidar_gt_path[pod_idx], i.replace('jpg', 'npy')) for i in test_list]
        paths['init_disp_path', pod_idx] = [os.path.join(args.init_disp_path[pod_idx], i.replace('jpg', 'npy')) for i in test_list]
        paths['pred_disp_path', pod_idx] = [os.path.join(args.pred_disp_path[pod_idx], i.replace('jpg', 'npy')) for i in test_list]
        paths['filename', pod_idx] = [i.split('.')[0] for i in test_list]


def parse_calibration_data(calib_path, inputs):
    with open(calib_path, "r") as f:
        calib_data = f.readlines()
        for line in calib_data:
            key, *values = line.split(':')
            if 'T' in key or 'bias' in key:
                inputs[key] = np.array(list(map(float, line.split(" ")[1:])))
            else:
                inputs[key] = np.array(list(map(float, line.split(" ")[1:]))).reshape(3, 3)


def remove_dust(disp_map, offset=1):
    # (x, y) 좌표에서 disparity gradient가 평균 gradient보다 크면
    # (x+offset, y), (x-offset, y) 좌표에
    # (x, y) 좌표의 disparity value를 적용시켜서
    # 3차원상에서 먼지같은 현상이 사라지도록 하는 함수

    avg_gradient_x = 30
    avg_gradient_y = 30
    # avg_gradient_x = np.mean(np.abs(disp_map[:, 1:] - disp_map[:, :-1]))
    # avg_gradient_y = np.mean(np.abs(disp_map[1:, :] - disp_map[:-1, :]))
    # grad_offset = avg_gradient / 1.5
    # avg_gradient = avg_gradient - grad_offset
    # print(avg_gradient)

    left_grad = np.abs(np.roll(disp_map, shift=-offset, axis=1) - disp_map)
    right_grad = np.abs(np.roll(disp_map, shift=offset, axis=1) - disp_map)
    up_grad = np.abs(np.roll(disp_map, shift=offset, axis=0) - disp_map)
    down_grad = np.abs(np.roll(disp_map, shift=offset, axis=0) - disp_map)

    left_mask = left_grad > avg_gradient_x
    right_mask = right_grad > avg_gradient_x
    up_mask = up_grad > avg_gradient_y
    down_mask = down_grad > avg_gradient_y

    disp_map[:, offset:][left_mask[:, offset:]] = disp_map[:, :-offset][left_mask[:, offset:]]
    # disp_map[:, :-offset][right_mask[:, :-offset]] = disp_map[:, offset:][right_mask[:, :-offset]]
    disp_map[offset:, :][up_mask[offset:, :]] = disp_map[:-offset, :][up_mask[offset:, :]]
    # disp_map[:-offset, :][down_mask[:-offset, :]] = disp_map[offset:, :][down_mask[:-offset, :]]

    return disp_map


def apply_icp(merge):
    source = merge[1]
    target = merge[0]
    # down sampling 해서 빠르게 매칭
    # source = source.voxel_down_sample(voxel_size=0.05)
    # target = target.voxel_down_sample(voxel_size=0.05)
    init_transformation = np.eye(4)
    # threshold is meter unit
    threshold = 0.01
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, init_transformation,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(), #)
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    return reg_p2p


def best_fit_transform(source, target):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    source = torch.Tensor(source.points).to('cuda')
    target = torch.Tensor(target.points).to('cuda')
    # get number of dimensions
    m = source.shape[1]

    # translate points to their centroids
    centroid_A = torch.mean(source, dim=0)
    centroid_B = torch.mean(target, dim=0)
    AA = source - centroid_A
    BB = target - centroid_B

    # rotation matrix
    H = AA.t().mm(BB)
    U, S, Vt = torch.svd(H)
    R = Vt.t().mm(U.t())

    # special reflection case
    if torch.det(R) < 0:
       Vt[m-1, :] *= -1
       R = Vt.t().mm(U.t())

    # translation
    t = centroid_B - centroid_A.mm(R)

    # homogeneous transformation
    T = torch.eye(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def visualize(args):
    paths = {}
    calib = {}

    # data path setting
    load_path(paths, args.n_pod)

    # visualization
    for file_idx in range(len(paths['filename', 0])):
        merge = []
        lidar_merge = []
        for n_pod in range(args.n_pod):
            img_l = o3d.io.read_image(paths['left_img_path', n_pod][file_idx])
            (height, width, _) = np.array(img_l).shape

            # calibration parameters setting
            parse_calibration_data(args.calib_path, calib)

            # Left POD
            if n_pod % 2 == 0:
                fx = calib['K_cam1'][0, 0]
                fy = calib['K_cam1'][1, 1]
                cx = calib['K_cam1'][0, 2]
                cy = calib['K_cam1'][1, 2]
                baseline = np.linalg.norm(calib['T_cam1_cam2'])
                # baseline = 0.367
                bias = calib['bias_offset_pod1']
            # Right POD
            else:
                fx = calib['K_cam3'][0, 0]
                fy = calib['K_cam3'][1, 1]
                cx = calib['K_cam3'][0, 2]
                cy = calib['K_cam3'][1, 2]
                baseline = np.linalg.norm(calib['T_cam3_cam4'])
                bias = calib['bias_offset_pod2']
                # baseline = 0.367
            intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

            if args.use_lidar:
                # if it is KITTI
                # lidar = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                # lidar_mask = lidar == 0
                # lidar[lidar_mask] = 1e-7
                # lidar_gt = (fx * baseline) / lidar
                # lidar_gt[lidar_mask] = 0
                lidar_gt = np.load(paths['lidar_gt_path', n_pod][file_idx]).astype(np.float32)
                lidar_gt = o3d.geometry.Image(lidar_gt)
                pcd_lidar = o3d.geometry.PointCloud.create_from_depth_image(lidar_gt, intrinsics)
                points = np.array(pcd_lidar.points).T
                if n_pod % 2 == 1:
                    points = calib['R_cam1_cam3'] @ points + calib['T_cam1_cam3'].reshape(3, 1)
                    points = o3d.utility.Vector3dVector(points.T)
                    pcd_lidar.points = points
                pcd_lidar.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            if args.use_init:
                disp = np.load(paths['init_disp_path', n_pod][file_idx])
            else:
                disp = np.load(paths['pred_disp_path', n_pod][file_idx]) * 700.
                # if it is KITTI
                # disp = cv2.imread(init_disp_path, cv2.IMREAD_GRAYSCALE)
                # disp = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE)

            print('\nRemove dust')
            disp = remove_dust(disp, offset=10)

            # depth = (fx * baseline) / (alpha * disp + beta)
            depth = (fx * baseline) / (bias[0] * disp + bias[1])
            depth = depth.clip(0, 10)
            depth_raw = o3d.geometry.Image(depth.astype(np.float32))
            if n_pod % 2 == 0:
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_l, depth_raw,
                                                                                depth_scale=1,
                                                                                depth_trunc=7,
                                                                                convert_rgb_to_intensity=False)
            else:
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_l, depth_raw,
                                                                                depth_scale=1,
                                                                                depth_trunc=7,
                                                                                convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

            if n_pod % 2 == 1:
                points = np.array(pcd.points).T
                XYZ = calib['R_cam1_cam3'] @ points + calib['T_cam1_cam3'].reshape(3, 1)
                XYZ = o3d.utility.Vector3dVector(XYZ.T)
                pcd.points = XYZ
                # extrinsics = np.hstack((calib['R_cam3_lidar'].T, calib['T_cam3_lidar'].reshape(3, 1)))
                # extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
                # pcd.transform(extrinsics)

            if args.use_filter:
                print('\nremove statistics outlier...')
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=3.0)
                pcd = pcd.select_by_index(ind)

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries([pcd])

            merge.append(pcd)
            lidar_merge.append(pcd_lidar)

            # ICP (lidar로 Transform 구하고 pred에 전달)
            if len(lidar_merge) > 1:
                print('ICP is on progress')
                reg_p2p = apply_icp(lidar_merge)
                # lidar_merge[1].transform(reg_p2p.transformation)
                # o3d.visualization.draw_geometries(lidar_merge)
                merge[1].transform(reg_p2p.transformation)
                # o3d.visualization.draw_geometries(merge)

            # if len(merge) > 1:
                # reg_p2p = apply_icp(merge)
                # merge[1].transform(reg_p2p.transformation)
                # o3d.visualization.draw_geometries(merge)

                # T, _, _ = best_fit_transform(merge[1], merge[0])
                # merge[1].transform(T)
                # o3d.visualization.draw_geometries(merge)

            # Visualize
            if args.vis:
                if args.use_lidar:
                    o3d.visualization.draw_geometries([pcd, pcd_lidar])
                else:
                    o3d.visualization.draw_geometries([pcd])
                if len(merge) > 1:
                    # points = np.asarray(merge[1].points).T
                    # points = calib['R_cam1_cam3'] @ (calib['R_cam3_cam4'] @ points - calib['T_cam3_cam4'].reshape(3, 1)) + calib['T_cam1_cam3'].reshape(3, 1)
                    # points = o3d.utility.Vector3dVector(points.T)
                    # merge[1].points = points
                    o3d.visualization.draw_geometries(merge)
                    o3d.visualization.draw_geometries(lidar_merge)
            # Save
            if args.save:
                if args.use_init:
                    name = 'init'
                else:
                    name = 'pred'
                o3d.io.write_point_cloud(os.path.join(args.save_path,
                                                      '{}_{}_{}.ply'.format(paths['filename', n_pod][file_idx], name,
                                                                            n_pod)), pcd)
                if args.use_lidar:
                    o3d.io.write_point_cloud(
                        os.path.join(args.save_path,
                                     '{}_{}_lidar.ply'.format(paths['filename', n_pod][file_idx], n_pod)),
                        pcd_lidar)
                if len(merge) > 1:
                    o3d.io.write_point_cloud(
                        os.path.join(args.save_path, '{}_merge.ply'.format(paths['filename', n_pod][file_idx])),
                        merge[0] + merge[1])
                print('save done')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.data_root = '/media/mino/SSD_8TB/DATASET/PO_data'
    args.dataset_name = '20230609'
    args.scene = 'scene_01'
    args.n_pod = 2

    if args.n_pod > 1:
        args.pods = ['stereo1_rect', 'stereo2_rect']
    else:
        args.pods = ['stereo1_rect']

    args.dataset_path = os.path.join(args.data_root, args.dataset_name, args.scene)
    args.save_path = os.path.join(args.dataset_path, 'ply')
    os.makedirs(args.save_path, exist_ok=True)
    args.list_path = os.path.join(args.dataset_path, 'train_list.txt')
    args.calib_path = os.path.join(args.data_root, args.dataset_name, 'calibration_params.txt')

    args.left_img_path = [os.path.join(args.dataset_path, pod, 'left') for pod in args.pods]
    args.lidar_gt_path = [os.path.join(args.dataset_path, pod, 'lidar_gt') for pod in args.pods]
    args.init_disp_path = [os.path.join(args.dataset_path, pod, 'unimatch', 'disp_left_np') for pod in args.pods]
    args.pred_disp_path = [os.path.join(args.dataset_path, pod, 'ADC') for pod in args.pods]

    args.use_lidar = True
    args.use_init = False
    args.use_filter = True

    args.save = True
    args.vis = False

    visualize(args)
