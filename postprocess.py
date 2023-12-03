from utils_numpy import flattenDetection, extract_patches, soft_argmax_2d, do_log
import numpy as np


SUPERPOINT_THRESHOLD = 0.05


def postprocess(semi, desc, nms_dist=4, conf_thresh=SUPERPOINT_THRESHOLD, patch_size=5):
    dn = np.linalg.norm(desc, ord=2, axis=1)
    desc = desc / np.expand_dims(dn, axis=1)

    output = {'semi': semi, 'desc': desc}
    heatmap = flattenDetection(semi)
    # nms
    heatmap_nms_batch = heatmap_to_nms(heatmap, nms_dist=nms_dist,
                                        conf_thresh=conf_thresh)
    outs = pred_soft_argmax(heatmap_nms_batch, heatmap, patch_size=patch_size)
    residual = outs['pred']
    # extract points
    outs = batch_extract_features(desc, heatmap_nms_batch, residual)
    output.update(outs)
    return output


def pred_soft_argmax(labels_2D, heatmap, patch_size):
    """
    return:
        dict {'loss': mean of difference btw pred and res}
    """
    outs = {}
    label_idx = np.stack(labels_2D.nonzero()).T
    patches = extract_patches(label_idx, heatmap, patch_size=patch_size)
    patches_log = do_log(patches)
    dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False)
    dxdy = dxdy.squeeze(1)
    dxdy = dxdy - patch_size // 2

    outs['pred'] = dxdy
    outs['patches'] = patches
    return outs


def sample_desc_from_points(coarse_desc, pts, cell_size=8):
    """
    inputs:
        coarse_desc: ndarray of shape (1, 256, Hc, Wc)
        pts: ndarray of shape (N, 2)
    return:
        desc: ndarray of shape (1, N, D)
    """

    def grid_sample(input, grid, padding_mode='zeros', align_corners=False):
        """
        Optimized implementation of torch.nn.functional.grid_sample for 4-D inputs.

        Args:
            input (ndarray): input of shape (N, C, H_in, W_in)
            grid (ndarray): flow-field of shape (N, H_out, W_out, 2)
            padding_mode (str): padding mode for outside grid values ('zeros' is supported)
            align_corners (bool): if True, consider the input pixels as squares

        Returns:
            output (ndarray): output ndarray
        """
        N, C, H_in, W_in = input.shape
        N, H_out, W_out, _ = grid.shape

        output = np.zeros((N, C, H_out, W_out))

        # Apply align_corners transformation
        if align_corners:
            grid = (grid + 1) * 0.5
            grid[..., 0] *= W_in - 1
            grid[..., 1] *= H_in - 1
        else:
            grid[..., 0] = (grid[..., 0] + 1) * 0.5 * (W_in - 1)
            grid[..., 1] = (grid[..., 1] + 1) * 0.5 * (H_in - 1)

        if padding_mode == 'zeros':
            # Handle out-of-bound grid values
            mask = (grid[..., 0] >= 0) & (grid[..., 0] <= W_in - 1) & (grid[..., 1] >= 0) & (grid[..., 1] <= H_in - 1)
            valid_grid = grid[mask]
            valid_indices = np.where(mask)

            # Interpolation indices
            x0 = np.floor(valid_grid[..., 0]).astype(int)
            y0 = np.floor(valid_grid[..., 1]).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1

            # Interpolation weights
            wx0 = valid_grid[..., 0] - x0
            wy0 = valid_grid[..., 1] - y0
            wx1 = 1 - wx0
            wy1 = 1 - wy0

            # Perform bilinear interpolation
            for c in range(C):
                output[valid_indices[0], c, valid_indices[1], valid_indices[2]] = (
                    wx1 * wy1 * input[valid_indices[0], c, y0, x0] +
                    wx0 * wy1 * input[valid_indices[0], c, y0, x1] +
                    wx1 * wy0 * input[valid_indices[0], c, y1, x0] +
                    wx0 * wy0 * input[valid_indices[0], c, y1, x1]
                )

        return output

    # --- Process descriptor.
    samp_pts = pts.T
    H, W = coarse_desc.shape[2] * cell_size, coarse_desc.shape[3] * cell_size
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.ones((1, 1, D))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.T.copy()
        samp_pts = samp_pts.reshape(1, 1, -1, 2)
        samp_pts = samp_pts.astype(float)

        desc = grid_sample(coarse_desc, samp_pts)
        desc = desc.squeeze(2).squeeze(0).T.reshape(1, -1, D)
    return desc


def heatmap_to_nms(heatmap, nms_dist, conf_thresh):
    """
    return:
      heatmap_nms_batch: np [batch, 1, H, W]
    """

    heatmap_nms_batch = [heatmap_nms(h, nms_dist, conf_thresh) for h in heatmap]
    heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
    heatmap_nms_batch = heatmap_nms_batch[:, np.newaxis, ...]
    return heatmap_nms_batch


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    border_remove = 4
    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
    """
    input:
        heatmap: np [(1), H, W]
    """
    heatmap = heatmap.squeeze()
    pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)

    semi_thd_nms_sample = np.zeros_like(heatmap)
    semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1

    return semi_thd_nms_sample


def batch_extract_features(desc, heatmap_nms_batch, residual):
    """
    return: -- type: dict
      desc: ndarray of shape (1, 256, 30, 40)
      heatmap_nms_batch: ndarray of shape (1, 1, 240, 320)
      residual: ndarray of shape (N, 2)
    """
    batch_size = heatmap_nms_batch.shape[0]

    pts_int, pts_desc = [], []
    pts_idx = np.argwhere(heatmap_nms_batch != 0)
    for i in range(batch_size):
        mask_b = (pts_idx[:, 0] == i)
        pts_int_b = pts_idx[mask_b][:, 2:].astype(float)
        pts_int_b = pts_int_b[:, [1, 0]]
        res_b = residual[mask_b]
        pts_b = pts_int_b + res_b
        pts_desc_b = sample_desc_from_points(desc[i].reshape(1, *desc[i].shape), pts_b).squeeze(0)
        pts_int.append(pts_int_b)
        pts_desc.append(pts_desc_b)

    pts_int = np.stack(pts_int, axis=0)
    pts_desc = np.stack(pts_desc, axis=0)
    return {'pts_int': pts_int, 'pts_desc': pts_desc}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle
    import cv2
    with open("shuttle.pkl", "rb") as f:
        semi, desc = pickle.load(f)
    out = postprocess(semi, desc)
    pts_int8 = out["pts_int"].squeeze()

    image = cv2.imread("space_shuttle_224.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))

    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap="gray")
    plt.scatter(pts_int8[:,0], pts_int8[:,1])
    plt.savefig("int8.jpg")


    from superpointnet import SuperPointNet
    import torch
    sp = SuperPointNet("/Users/topkech/Work/satnav/visual_navigation_satellite/data/models/superPointNet_114000_checkpoint.pth.tar")
    sp.eval()
    semi, desc = sp(torch.tensor(image[np.newaxis, np.newaxis, ...]/255, dtype=torch.float32))
    out = postprocess(semi.detach().numpy(), desc.detach().numpy())
    pts_fp32 = out["pts_int"].squeeze()

    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap="gray")
    plt.scatter(pts_fp32[:,0], pts_fp32[:,1])
    plt.savefig("fp32.jpg")
