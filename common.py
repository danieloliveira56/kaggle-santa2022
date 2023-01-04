import itertools
import numpy as np
import pandas as pd
import datetime
from numba import njit
from rich.progress import track

INITIAL_CONFIG = np.array(
    [[64, 0],
    [-32, 0],
    [-16, 0],
    [-8, 0],
    [-4, 0],
    [-2, 0],
    [-1, 0],
    [-1, 0]]
)
ARM_LENGTHS = {
    0: 64,
    1: 32,
    2: 16,
    3: 8,
    4: 4,
    5: 2,
    6: 1,
    7: 1,
}
IMAGE_LUT = (pd.read_csv("image.csv") + np.array([[128, 128, 0, 0, 0]])).to_numpy()
image = np.zeros(shape=(257, 257, 3))
for x, y, r, g, b in IMAGE_LUT:
    image[int(x), int(y)] = [r, g, b]

@njit
def pos2lut_idx(pos):
    """Convert positions in the range of [-128, 128] into row index for the RGB-LUT"""
    transformed_pos = pos + 128
    return transformed_pos[:, 0] + (256 - transformed_pos[:, 1])*257

@njit
def cost_fun(config, rgb_path):
    """This cost function takes the configuration matrix and the corresponding visited
    colors of the path as input and returns the scalar float cost"""
    return np.sqrt(np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)).sum() +\
                  (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum())

@njit
def evaluate_solution(config):
    """Generates the RGB-path from the configuration matrix and calls the cost function"""
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    return cost_fun(config, rgb_path)

@njit
def cost(config1, config2):
    x1, y1 = np.sum(config1, axis=0)
    x2, y2 = np.sum(config2, axis=0)

    return np.sqrt(np.abs(config1 - config2).sum()) \
        + 3.0 * (np.abs(image[x1+128, y1+128] - image[x2+128, y2+128]).sum())


def cost_array(config):
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    return np.sqrt(np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)) +\
                      (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum(axis=-1))

def reference_config_cost(reference_config, config_array):
    print(reference_config.shape)
    print(config_array.shape)
    n = config_array.shape[0]
    config_array2 = np.tile(reference_config, (n, 1, 1))
    # config_array2 = np.repeat(reference_config, n, axis=0).reshape((n, 8, 2), order='F')

    print(config_array2)
    print(config_array)
    return evaluate_relative_cost(config_array2, config_array)


# @njit
def evaluate_relative_cost(config_array1, config_array2):
    lut_idx = pos2lut_idx(config_array1.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    lut_idx2 = pos2lut_idx(config_array2.sum(axis=1))
    rgb_path2 = IMAGE_LUT[lut_idx2, -3:]

    cost = np.sqrt(np.abs(config_array2 - config_array1).sum(axis=-1).sum(axis=-1)) + \
           (3.0 * np.abs(rgb_path2 - rgb_path).sum(axis=-1))
    reconfig_change = np.abs(config_array2 - config_array1).max(axis=-1).sum(axis=-1)
    cost[reconfig_change > 1] = 1e2
    return cost


def is_valid(i, j):
    return np.abs(i - j).max(axis=-1).sum(axis=-1) <= 1


def config_to_string(config):
    return ';'.join([' '.join(map(str, vector)) for vector in config])


def solution_to_submission(solution, save_as='submission'):
    submission = pd.Series(
    [config_to_string(config) for config in solution],
    name="configuration",
    )
    submission.to_csv(f"{save_as}-{datetime.datetime.now():%h%d-%H%M}.csv", index=False)


def config_to_string(config):
    return ';'.join([' '.join(map(str, vector)) for vector in config])


def load_submission(csv_filename):
    # read sample submission, transform to 3D-Tensor
    csv_df = pd.read_csv(csv_filename, skiprows=1, sep=';', header=None)
    submission = np.array([[r.split(' ') for r in row] for _, row in csv_df.iterrows()], dtype=int)
    print(f"Loaded solution from {csv_filename}\n\tLength: {submission.shape}\n\tCost: {evaluate_solution(submission)}")
    return submission


def solution_to_xy_config_dict(solution):
    xy_config_dict = {
        (i, j): []
        for i in range(-128, 129)
        for j in range(-128, 129)
    }
    for i, config in enumerate(solution):
        xy_config_dict[tuple(sum(config))].append((i, config))
    return xy_config_dict


def adjacency_dict(config_pool):
    xy_config_dict = solution_to_xy_config_dict(config_pool)
    print("xy_config_dict created")
    N = {
        np.array2string(i): np.empty((0, 8, 2), dtype=int)
        for i in config_pool
    }
    print("empty N created")

    for i in track(config_pool, description="Creating adjacency_dict"):
        xi,yi = i.sum(axis=0)
        for delta_x in range(-8, 9):
            for delta_y in range(-8+abs(delta_x), 9-abs(delta_x)):
                xj = xi+delta_x
                if xj < -128 or xj > 128:
                    continue
                yj = yi+delta_y
                if yj < -128 or yj > 128:
                    continue
                for _, j in xy_config_dict[(xj, yj)]:
                    if not is_valid(i, j):
                        continue
                    N[np.array2string(i)] = np.unique(np.vstack([N[np.array2string(i)], j.reshape(1, 8, 2)]), axis=0)
                    N[np.array2string(j)] = np.unique(np.vstack([N[np.array2string(j)], i.reshape(1, 8, 2)]), axis=0)
    return N


def cost_map(config_pool):
    N = adjacency_dict(config_pool)

    return {
        np.array2string(i): {
            np.array2string(j): cost
            for j, cost in zip(N[np.array2string(i)], reference_config_cost(i, N[np.array2string(i)]))
        }
        for i in config_pool
    }


def get_neighbors(config):
    valid_configs = [
        [config[pos]]
        for pos in range(8)
    ]
    for pos in range(8):
        for x_step, y_step in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            if abs(config[pos][0] + x_step) != ARM_LENGTHS[pos] and abs(config[pos][1] + y_step) != ARM_LENGTHS[pos]:
                continue
            if abs(config[pos][0] + x_step) > ARM_LENGTHS[pos] or abs(config[pos][1] + y_step) > ARM_LENGTHS[pos]:
                continue
            valid_configs[pos].append(np.array((config[pos][0] + x_step, config[pos][1] + y_step)))

    for config2 in itertools.product(
        valid_configs[0],
        valid_configs[1],
        valid_configs[2],
        valid_configs[3],
        valid_configs[4],
        valid_configs[5],
        valid_configs[6],
        valid_configs[7]
    ):
        config2 = np.array(config2)
        cost = np.abs(config2-config).sum()
        dist = np.abs(np.sum(config2, axis=0)-np.sum(config, axis=0)).sum()
        if cost <= dist:
            yield config2, cost


def shortest_path(source, sink):
    Q = [(source, 0)]
    cost_table = {}
    while Q:
        V, cur_cost = Q.pop()
        for n, cost in get_neighbors(V):
            xy = tuple(np.sum(n, axis=0))
            if xy not in cost_table or cur_cost + cost < cost_table[xy]:
                cost_table[xy] = cur_cost + cost
                Q.append((n, cur_cost + cost))
        Q = sorted(Q, key=lambda q: -q[1])
        print(len(Q))
    return cost_table[sink]


def plot_traj(points, image):
    origin = np.array([0, 0])
    lines = []
    if not (origin == points[0]).all():
        lines.append([origin, points[0]])
    for i in range(1, len(points)):
        lines.append([points[i - 1], points[i]])
    if not (origin == points[1]).all():
        lines.append([points[-1], origin])

    colors = []
    for l in lines:
        dist = np.abs(l[0] - l[1]).max()
        if dist <= 2:
            colors.append('b')
        else:
            colors.append('r')

    lc = mc.LineCollection(lines, colors=colors)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    radius = image.shape[0] // 2
    ax.matshow(image * 0.8 + 0.2, extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5))
    ax.grid(None)

    ax.autoscale()
    fig.show()