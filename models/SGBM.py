import torch.nn as nn
import cv2


# traditional stereo matching (Stereo Semi-Global Block Matching)
class SGBM(nn.Module):
    def __init__(self):
        super().__init__()
        window_size = 5
        min_disp = 16
        num_disp = 112 - min_disp
        self.stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                            numDisparities=num_disp,
                                            blockSize=window_size,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            disp12MaxDiff=1,
                                            P1=8 * 3 * window_size ** 2,
                                            P2=32 * 3 * window_size ** 2)

    def forward(self, inputs, idx):
        # Compute the disparity map
        disparity = self.stereo.compute(inputs['img_l', idx], inputs['img_r', idx]).astype(np.float32) / 16.0

        # save

        # Normalize the disparity for visualization
        # disparity_normalized = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
        # disparity_normalized = np.uint8(disparity_normalized)
        # Display the disparity map
        # cv2.imshow('Disparity', disparity_normalized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()