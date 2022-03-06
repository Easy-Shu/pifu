import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    # normalize range from [0, res] to [b_min, b_max]
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool_)  # dirty means not evaluated points
    grid_mask = np.zeros(resolution, dtype=np.bool_)

    eval_res = resolution[0] // init_resolution  # sample step

    while eval_res > 0:  # sample from coarse to fine
        # sample the grid sparsely, sample step = eval_res
        grid_mask[0:resolution[0]:eval_res, 0:resolution[1]:eval_res, 0:resolution[2]:eval_res] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        # print('step size:', eval_res, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        # if border points of a region share similar sdf values, interpolate region with mean value
        if eval_res <= 1:
            break
        for x in range(0, resolution[0] - eval_res, eval_res):
            for y in range(0, resolution[1] - eval_res, eval_res):
                for z in range(0, resolution[2] - eval_res, eval_res):
                    # if center marked, return
                    if not dirty[x + eval_res // 2, y + eval_res // 2, z + eval_res // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + eval_res]
                    v2 = sdf[x, y + eval_res, z]
                    v3 = sdf[x, y + eval_res, z + eval_res]
                    v4 = sdf[x + eval_res, y, z]
                    v5 = sdf[x + eval_res, y, z + eval_res]
                    v6 = sdf[x + eval_res, y + eval_res, z]
                    v7 = sdf[x + eval_res, y + eval_res, z + eval_res]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + eval_res, y:y + eval_res, z:z + eval_res] = (v_max + v_min) / 2
                        dirty[x:x + eval_res, y:y + eval_res, z:z + eval_res] = False
        eval_res //= 2

    return sdf.reshape(resolution)
