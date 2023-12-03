from typing import Tuple
import numpy as np
import torch
import cv2
import scipy

from .spatial_soft_argmax2d_np import SpatialSoftArgmax2d

SUPERPOINT_EROSION_KSIZE = 3

class DepthToSpace:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = np.transpose(input, (0, 2, 3, 1))
        (batch_size, d_height, d_width, d_depth) = output.shape
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = np.split(t_1, self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = np.concatenate(stack,0)
        output = np.expand_dims(output, 1)

        output = np.transpose(output, (1, 0, 2, 3, 4))

        output = np.transpose(output, (0, 2, 1, 3, 4))
        output = output.reshape(batch_size, s_height, s_width, s_depth)
        output = np.transpose(output, (0, 3, 1, 2))

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def flattenDetection(semi, tensor=False):
    '''
    Flatten detection output

    :param semi:
        output from detector head
        ndarray [65, Hc, Wc]
        :or
        ndarray (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        ndarray (batch_size, 65, Hc, Wc)

    '''
    batch = False
    if len(semi.shape) == 4:
        batch = True

    if batch:
        dense = scipy.special.softmax(semi, axis=1) # [batch, 65, Hc, Wc]

        nodust = dense[:, :-1, :, :]
    else:
        dense = scipy.special.softmax(semi, axis=0) # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)
    # Reshape to get full resolution heatmap.
    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)
    heatmap = np.squeeze(heatmap, 0) if not batch else heatmap
    return heatmap


def pts_to_bbox(points, patch_size):
    """
    input:
        points: (y, x)
    output:
        bbox: (x1, y1, x2, y2)
    """

    shift_l = (patch_size+1) / 2
    shift_r = patch_size - shift_l
    pts_l = points-shift_l
    pts_r = points+shift_r+1
    bbox = np.stack((pts_l[:, 1], pts_l[:, 0], pts_r[:, 1], pts_r[:, 0]), axis=1)

    return bbox


def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    pred_heatmap = torch.Tensor(pred_heatmap).cpu()
    rois = torch.Tensor(rois).cpu()
    patches = roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
    patches = patches.detach().cpu().numpy()
    return patches


def extract_patches(label_idx, image, patch_size=7):
    """
    return:
        patches: ndarray [N, 1, patch, patch]
    """
    rois = pts_to_bbox(label_idx[:,2:], patch_size).astype(np.int64)
    rois = np.concatenate((label_idx[:,:1], rois), axis=1)
    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches


def soft_argmax_2d(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """

    m = SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)

    if patches.shape[0] > 0:
        coords = m(patches)
    else:
        coords = np.zeros([0, 1, 2])
    return coords


def do_log(patches):
    patches[patches < 0] = 1e-6
    patches_log = np.log(patches)
    return patches_log


def filter_kp_by_masked_image(image: np.ndarray, keypoints: np.ndarray, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts zeros mask from image and filters out points that are not included in this mask

    Args:
        image (np.ndarray): image in shape [1, 1, W, H]
        keypoints (np.ndarray): keypoints in shape [1, N, 2], where N - number of points
        descriptors (np.ndarray): descriptors in shape [1, N, L], where L - descriptor's length

    Returns:
        Tuple[np.ndarray, np.ndarray]: filtered keypoints and descriptors in shapes [1, M, 2] & [1, M, L],
            where M - number of points inside non-zero mask of the image
    """
    image = image.astype(np.float32)
    keypoints = keypoints.astype(np.float32)
    descriptors = descriptors.astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SUPERPOINT_EROSION_KSIZE, SUPERPOINT_EROSION_KSIZE))

    mask = (image > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask.squeeze(), cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=3)

    cols = keypoints[:, 0].astype(np.int32)
    rows = keypoints[:, 1].astype(np.int32)
    kp_mask = mask[rows, cols] > 0

    keypoints = keypoints[kp_mask, :]
    descriptors = descriptors[kp_mask, :]
    return keypoints, descriptors
