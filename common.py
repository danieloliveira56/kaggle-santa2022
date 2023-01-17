import datetime
import itertools

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from rich.progress import track

plt.style.use("seaborn-whitegrid")

TOP_RIGHT = 0
TOP_LEFT = 1
BOTTOM_LEFT = 2
BOTTOM_RIGHT = 3
AXIS = 4

INITIAL_CONFIG = np.array(
    [[64, 0], [-32, 0], [-16, 0], [-8, 0], [-4, 0], [-2, 0], [-1, 0], [-1, 0]], dtype=int
)
ARM_LENGTHS = np.array([1, 1, 2, 4, 8, 16, 32, 64], dtype=int)

IMAGE_LUT = (pd.read_csv("image.csv") + np.array([[128, 128, 0, 0, 0]])).to_numpy()
image = np.zeros(shape=(257, 257, 3))
for x, y, r, g, b in IMAGE_LUT:
    image[int(x), int(y)] = [r, g, b]


@njit
def pos2lut_idx(pos):
    """Convert positions in the range of [-128, 128] into row index for the RGB-LUT"""
    transformed_pos = pos + 128
    return transformed_pos[:, 0] + (256 - transformed_pos[:, 1]) * 257


@njit
def cost_fun(config, rgb_path):
    """This cost function takes the configuration matrix and the corresponding visited
    colors of the path as input and returns the scalar float cost"""
    return np.sqrt(
        np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)
    ).sum() + (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum())


@njit
def evaluate_solution(config):
    """Generates the RGB-path from the configuration matrix and calls the cost function"""
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    return cost_fun(config, rgb_path)


@njit
def cost(config1, config2, duplicate_xy=False):
    x1, y1 = np.sum(config1, axis=0)
    x2, y2 = np.sum(config2, axis=0)
    if duplicate_xy and x1 == x2 and y1 == y2:
        return 0

    return np.sqrt(np.abs(config1 - config2).sum()) + 3.0 * (
        np.abs(image[x1 + 128, y1 + 128] - image[x2 + 128, y2 + 128]).sum()
    )

@njit
def cost_xy(x1, y1, x2, y2):
    cost = abs(x1-x2) + abs(y1-y2)

    return np.sqrt(cost) + 3.0 * (np.abs(image[x1 + 128, y1 + 128] - image[x2 + 128, y2 + 128]).sum())


def cost_array(config):
    lut_idx = pos2lut_idx(config.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    return np.sqrt(
        np.abs(config[:-1, :, :] - config[1:, :, :]).sum(axis=-1).sum(axis=-1)
    ) + (3.0 * np.abs(rgb_path[:-1, :] - rgb_path[1:, :]).sum(axis=-1))


def reference_config_cost(reference_config, config_array):
    print(reference_config.shape)
    print(config_array.shape)
    n = config_array.shape[0]
    config_array2 = np.tile(reference_config, (n, 1, 1))
    # config_array2 = np.repeat(reference_config, n, axis=0).reshape((n, 8, 2), order='F')

    print(config_array2)
    print(config_array)
    return evaluate_relative_cost(config_array2, config_array)


def print_solution(solution):
    for i, config in enumerate(solution):
        print(
            f"{i+1} ({cost(solution[i-1], solution[i]) if i > 0 else 0:.2f}):",
            np.array2string(config)
            .replace("\n", ";")
            .replace("[[", "[")
            .replace("]]", "]"),
            config.sum(axis=0),
        )


# @njit
def evaluate_relative_cost(config_array1, config_array2):
    lut_idx = pos2lut_idx(config_array1.sum(axis=1))
    rgb_path = IMAGE_LUT[lut_idx, -3:]
    lut_idx2 = pos2lut_idx(config_array2.sum(axis=1))
    rgb_path2 = IMAGE_LUT[lut_idx2, -3:]

    cost = np.sqrt(np.abs(config_array2 - config_array1).sum(axis=-1).sum(axis=-1)) + (
        3.0 * np.abs(rgb_path2 - rgb_path).sum(axis=-1)
    )
    reconfig_change = np.abs(config_array2 - config_array1).max(axis=-1).sum(axis=-1)
    cost[reconfig_change > 1] = 1e2
    return cost


def is_valid(i, j):
    return np.abs(i - j).max(axis=-1).max() == 1


def valid_config(i):
    return np.all(np.abs(i).max(axis=-1) == ARM_LENGTHS)


def config_to_string(config):
    return ";".join([" ".join(map(str, vector)) for vector in config])


def solution_to_submission(solution, save_as="submission"):
    submission = pd.Series(
        [config_to_string(config) for config in solution],
        name="configuration",
    )
    submission.to_csv(
        f"{save_as}-{10*evaluate_solution(solution):.0f}-{solution.shape[0]}-{validate_solution(solution)}-{datetime.datetime.now():%h%d-%H%M}.csv",
        index=False,
    )


def config_to_string(config):
    return ";".join([" ".join(map(str, vector)) for vector in config])


def load_submission(csv_filename):
    # read sample submission, transform to 3D-Tensor
    csv_df = pd.read_csv(csv_filename, skiprows=1, sep=";", header=None)
    submission = np.array(
        [[r.split(" ") for r in row] for _, row in csv_df.iterrows()], dtype=int
    )
    print(
        f"Loaded solution from {csv_filename}\n\tLength: {submission.shape}\n\tCost: {evaluate_solution(submission)}"
    )
    return submission


def delete_indices(config_pool, to_delete):
    print(f"Deleting {len(to_delete)} indices: {to_delete}")
    print(config_pool.shape)
    to_delete = sorted(to_delete, reverse=True)
    for i in to_delete:
        config_pool = np.concatenate([config_pool[:i], config_pool[i + 1 :]])
    print(config_pool.shape)
    return config_pool


def solution_to_xy_config_dict(config_pool):
    xy_config_dict = {(x, y): set() for x in range(-128, 129) for y in range(-128, 129)}
    for i, config in track(
        enumerate(config_pool), description="Creating xy_config dict"
    ):
        xy_config_dict[tuple(sum(config))].add(i)
    return xy_config_dict


def validate_solution(solution):
    invalid_edges = 0

    pts_seen = set()
    pts_seen.add(tuple(solution[0].sum(axis=0)))
    for i in range(1, solution.shape[0]):
        config1 = solution[i - 1]
        config2 = solution[i]
        if not is_valid(config1, config2):
            print(f"Invalid move {i-1}->{i}, Cost {cost(config1, config2)}:")
            invalid_edges += 1
            print(
                np.array2string(config1)
                .replace("\n", ";")
                .replace("[[", "[")
                .replace("]]", "]"),
                np.array2string(config1.sum(axis=0))
                .replace("\n", ";")
                .replace("[[", "[")
                .replace("]]", "]"),
            )
            print("->")
            print(
                np.array2string(config2)
                .replace("\n", ";")
                .replace("[[", "[")
                .replace("]]", "]"),
                np.array2string(config2.sum(axis=0))
                .replace("\n", ";")
                .replace("[[", "[")
                .replace("]]", "]"),
            )
            print(
                np.array2string(config2 - config1)
                .replace("\n", ";")
                .replace("[[", "[")
                .replace("]]", "]")
            )
            print()
            valid = False
        pts_seen.add(tuple(config2.sum(axis=0)))

    num_missing_pts = 257 * 257 - len(pts_seen)
    if num_missing_pts:
        print("Not visited all points")

    return (invalid_edges, num_missing_pts)


def compress_solution(solution):
    was_compressed = True
    while was_compressed:
        was_compressed = False
        for i in range(solution.shape[0] - 2):
            for j in range(i + 2, solution.shape[0]):
                if not is_valid(solution[i], solution[j]):
                    continue
                print(i, j)
                visited = set()
                for config in solution[:i]:
                    visited.add(tuple(config.sum(axis=0)))
                for config in solution[j + 1 :]:
                    visited.add(tuple(config.sum(axis=0)))

                compressible = True
                for config in solution[i : j + 1]:
                    if tuple(config.sum(axis=0)) not in visited:
                        compressible = False
                        break
                if compressible:
                    solution = np.concatenate([solution[:i], solution[j + 1 :]])
                    was_compressed = True
                    print(
                        f"Solution compressed, {solution.shape}, {evaluate_solution(solution)}"
                    )
                    break
            if was_compressed:
                break
    return solution


def adjacency_dict(config_pool, same_xy_is_adjacent=False):
    xy_config_dict = solution_to_xy_config_dict(config_pool)
    N = {i: set() for i in range(config_pool.shape[0])}
    for i, ci in track(enumerate(config_pool), description="Populating adjacency dict"):
        xi, yi = ci.sum(axis=0)
        for delta_x in range(-8, 9):
            for delta_y in range(-8 + abs(delta_x), 9 - abs(delta_x)):
                xj = xi + delta_x
                if xj < -128 or xj > 128:
                    continue
                yj = yi + delta_y
                if yj < -128 or yj > 128:
                    continue
                for j in xy_config_dict[(xj, yj)]:
                    if is_valid(config_pool[i], config_pool[j]) or (
                        same_xy_is_adjacent and delta_x == 0 and delta_y == 0
                    ):
                        N[i].add(j)
                        N[j].add(i)
    return N


def cost_map(config_pool, duplicate_xy=False):
    N = adjacency_dict(config_pool, same_xy_is_adjacent=duplicate_xy)
    return {
        i: {j: cost(ci, config_pool[j], duplicate_xy) for j in N[i]}
        for i, ci in enumerate(config_pool)
    }


# Deprecated
def get_neighbors(config, max_dist=8):
    valid_configs = [[config[pos]] for pos in range(8)]
    for pos in range(8):
        for x_step, y_step in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            if (
                abs(config[pos][0] + x_step) != ARM_LENGTHS[pos]
                and abs(config[pos][1] + y_step) != ARM_LENGTHS[pos]
            ):
                continue
            if (
                abs(config[pos][0] + x_step) > ARM_LENGTHS[pos]
                or abs(config[pos][1] + y_step) > ARM_LENGTHS[pos]
            ):
                continue
            valid_configs[pos].append(
                np.array((config[pos][0] + x_step, config[pos][1] + y_step))
            )

    for config2 in itertools.product(
        valid_configs[0],
        valid_configs[1],
        valid_configs[2],
        valid_configs[3],
        valid_configs[4],
        valid_configs[5],
        valid_configs[6],
        valid_configs[7],
    ):
        config2 = np.array(config2)
        dist = np.abs(config2 - config).sum()
        if is_valid(config, config2) and dist != 0 and dist <= max_dist:
            yield config2


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
            colors.append("b")
        else:
            colors.append("r")

    lc = mc.LineCollection(lines, colors=colors)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    radius = image.shape[0] // 2
    ax.matshow(
        image * 0.8 + 0.2,
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
    )
    ax.grid(None)

    ax.autoscale()
    fig.show()


def enumerate_configs():
    valid_configs = [
        [[x, y] for x in [-l, l] for y in range(-l, l + 1)]
        + [[x, y] for x in range(-l + 1, l) for y in [-l, l]]
        for l in [64, 32, 16, 8, 4, 2, 1, 1]
    ]
    for config in itertools.product(
        valid_configs[0],
        valid_configs[1],
        valid_configs[2],
        valid_configs[3],
        valid_configs[4],
        valid_configs[5],
        valid_configs[6],
        valid_configs[7],
    ):
        yield np.array(config)


def enumerate_quadrants():
    valid_configs = [
        [[x, y] for x in range(-l, l + 1) for y in [-l, l]]
        for l in [32, 16, 8, 4, 2, 1, 1]
    ]
    valid_configs = [
        [[x, y] for x in [-64, 64] for y in range(-64, 65)]
    ] + valid_configs
    for config in itertools.product(
        valid_configs[0],
        valid_configs[1],
        valid_configs[2],
        valid_configs[3],
        valid_configs[4],
        valid_configs[5],
        valid_configs[6],
        valid_configs[7],
    ):
        yield np.array(config)


def sign(x):
    return -1 if x < 0 else 1


# def standard_config(x, y):
#     """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
#     xr = sign(x) * 64
#     yr = sign(y) * 64
#     config = [(xr, y - yr)]
#     x -= xr
#     while abs(xr) > 1:
#         xr //= 2
#         yr //= 2
#         cur_x = np.clip(x + xr, -abs(xr), abs(xr))
#         config.append((cur_x, yr))
#         x -= cur_x
#     config.append((x, yr))
#
#     return np.array(config)


def standard_config_topright(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x > 0 and y >= 0, "This function is only for the upper right quadrant"
    r = 64
    config = [(r, y - r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, r))  # arm points upwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, r))  # arm points upwards
    assert x == arm_x
    return np.array(config)


def standard_config_bottomright(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x >= 0 and y < 0, "This function is only for the bottom right quadrant"
    r = 64
    config = [(x - r, -r)]  # longest arm points to the right
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        arm_y = np.clip(y, -r, r)
        config.append((r, arm_y))  # arm points upwards
        y -= arm_y
    arm_y = np.clip(y, -r, r)
    config.append((r, arm_y))  # arm points upwards
    assert y == arm_y
    return np.array(config)


def standard_config_topleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x <= 0 and y > 0, "This function is only for the upper left quadrant"
    # (_, 64), (-32, _), (-16, _) ,(-8, _), (-4, _), (-2, _), (-1, _), (-1, _)
    r = 64
    config = [(x - (-r), r)]  # longest arm points to the top
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        arm_y = np.clip(y, -r, r)
        config.append((-r, arm_y))  # arm points leftwards
        y -= arm_y
    arm_y = np.clip(y, -r, r)
    config.append((-r, arm_y))  # arm points leftwards
    assert y == arm_y
    return np.array(config)


def standard_config_bottomleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    assert x < 0 and y <= 0, "This function is only for the lower left quadrant"
    # (-64, _),(_, -32), (_, -16) ,(_, -8), (_, -4), (_, -2), (_, -1), (_, -1)
    r = 64
    config = [(-r, y - (-r))]  # longest arm points to the left
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points downwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points downwards
    assert x == arm_x
    return np.array(config)


def quadrant_pts_topright():
    for x in range(1, 129):
        for y in range(129):
            yield x, y


def quadrant_pts_topleft():
    for x in range(-128, 1):
        for y in range(1, 129):
            yield x, y


def quadrant_pts_bottomleft():
    for x in range(-128, 0):
        for y in range(-128, 1):
            yield x, y


def quadrant_pts_bottomright():
    for x in range(129):
        for y in range(-128, 0):
            yield x, y


quadrant_pts = {
    TOP_RIGHT: quadrant_pts_topright(),
    TOP_LEFT: quadrant_pts_topleft(),
    BOTTOM_LEFT: quadrant_pts_bottomleft(),
    BOTTOM_RIGHT: quadrant_pts_bottomright(),
}

standard_config = {
    TOP_RIGHT: standard_config_topright,
    TOP_LEFT: standard_config_topleft,
    BOTTOM_LEFT: standard_config_bottomleft,
    BOTTOM_RIGHT: standard_config_bottomright,
}


def get_axis_configs():
    return np.concatenate(
        [
            np.array(
                [
                    [
                        (64, max(0, y - 64)),
                        (-32, np.clip(y - 32, 0, 32)),
                        (-16, np.clip(y - 16, 0, 16)),
                        (-8, np.clip(y - 8, 0, 8)),
                        (-4, np.clip(y - 4, 0, 4)),
                        (-2, np.clip(y - 2, 0, 2)),
                        (-1, np.clip(y - 1, 0, 1)),
                        (-1, np.clip(y, 0, 1)),
                    ]
                    for y in range(0, 129)
                ]
            ),
            np.array(
                [
                    [
                        (64, -max(0, y - 64)),
                        (-32, -np.clip(y - 32, 0, 32)),
                        (-16, -np.clip(y - 16, 0, 16)),
                        (-8, -np.clip(y - 8, 0, 8)),
                        (-4, -np.clip(y - 4, 0, 4)),
                        (-2, -np.clip(y - 2, 0, 2)),
                        (-1, -np.clip(y - 1, 0, 1)),
                        (-1, -np.clip(y, 0, 1)),
                    ]
                    for y in range(1, 129)
                ]
            ),
        ]
    )


def get_quadrant_solution(quadrant):
    quadrant_solution = []
    for x, y in quadrant_pts[quadrant]:
        config = standard_config[quadrant](x, y)
        assert np.abs(config[0]).max() == 64
        assert np.abs(config[1]).max() == 32
        assert np.abs(config[2]).max() == 16
        assert np.abs(config[3]).max() == 8
        assert np.abs(config[4]).max() == 4
        assert np.abs(config[5]).max() == 2
        assert np.abs(config[6]).max() == 1
        assert np.abs(config[7]).max() == 1

        x_config, y_config = config.sum(axis=0)
        assert x_config == x
        assert y_config == y
        quadrant_solution.append(config)

    quadrant_solution = np.array(quadrant_solution)
    print(quadrant_solution.shape)
    quadrant_solution = np.concatenate(
        [
            # np.array([INITIAL_CONFIG]),
            quadrant_solution
            # np.array([INITIAL_CONFIG])
        ]
    )
    return quadrant_solution


def get_quadrant_path():
    topright_path = [(x, 0) for x in range(1, 129)] + [(128, y) for y in range(1, 129)]
    x = 127
    y = 128
    y_direction = -1
    while x > 0:
        while y >= 1 and y <= 128:
            topright_path.append((x, y))
            y += y_direction
        y_direction = -y_direction
        x -= 1
        y += y_direction

    topleft_path = [(x, 1) for x in range(0, -129, -1)] + [
        (-128, y) for y in range(2, 129)
    ]
    x, y = topleft_path[-1]
    x_direction = -sign(x)
    x += x_direction
    while y > 0:
        while x >= -128 and x <= 0:
            topleft_path.append((x, y))
            x += x_direction
        x_direction = -x_direction
        x += x_direction
        y -= 1

    bottomleft_path = [(x, 0) for x in range(-1, -129, -1)] + [
        (-128, y) for y in range(-1, -129, -1)
    ]
    x, y = bottomleft_path[-1]
    x_direction = -sign(x)
    y_direction = -sign(y)
    x += x_direction
    while y <= 0:
        while x >= -128 and x < 0:
            bottomleft_path.append((x, y))
            x += x_direction
        x_direction = -x_direction
        x += x_direction
        y += y_direction

    bottomright_path = [(0, y) for y in range(-1, -129, -1)] + [
        (x, -128) for x in range(1, 129)
    ]
    x, y = bottomright_path[-1]
    x_direction = -sign(x)
    y_direction = -sign(y)
    x += x_direction
    while y < 0:
        while x >= 0 and x <= 128:
            bottomright_path.append((x, y))
            x += x_direction
        x_direction = -x_direction
        x += x_direction
        y += y_direction

    solution = np.concatenate(
        [
            np.array([INITIAL_CONFIG]),
            [standard_config[TOP_RIGHT](x, y) for x, y in topright_path],
            np.array([INITIAL_CONFIG]),
            [standard_config[TOP_LEFT](x, y) for x, y in topleft_path],
            np.array([INITIAL_CONFIG]),
            [standard_config[BOTTOM_LEFT](x, y) for x, y in bottomleft_path],
            np.array([INITIAL_CONFIG]),
            [standard_config[BOTTOM_RIGHT](x, y) for x, y in bottomright_path],
            np.array([INITIAL_CONFIG]),
        ]
    )

    print(solution.shape)
    print(evaluate_solution(solution))
    validate_solution(solution)

    solution_to_submission(solution)


def remove_replacements(solution):
    # Remove replacements
    i = 0
    print("Checking for replaceable configs (Set TSP)", solution.shape)
    while i < solution.shape[0]:
        entry_config = i
        x, y = solution[i].sum(axis=0)
        while i + 1 < solution.shape[0] and np.array_equal(
            solution[i + 1].sum(axis=0), np.array((x, y))
        ):
            i += 1
        if entry_config < i and is_valid(solution[entry_config], solution[i + 1]):
            # Found valid replacement, configs entry_config + 1, ..., i can be deleted
            print(f"Replacing configs {entry_config}...{i - 1}")
            for j in range(entry_config - 1, i + 3):
                print(
                    j,
                    config_to_string(solution[j]),
                    solution[j].sum(axis=0),
                )
            if is_valid(solution[entry_config - 1], solution[entry_config]):
                solution = np.concatenate(
                    [solution[: entry_config + 1], solution[i + 1 :]]
                )
            else:
                solution = np.concatenate([solution[:entry_config], solution[i:]])

            print("---")
            for j in range(entry_config - 1, i + 3):
                print(
                    j,
                    np.array2string(solution[j])
                    .replace("\n", ";")
                    .replace("[", "")
                    .replace("]", ""),
                    solution[j].sum(axis=0),
                )
            print(solution.shape)
        else:
            i += 1
    return solution


def num_configs():
    num_configs = 1
    for l in [64, 32, 16, 8, 4, 2, 1, 1]:
        num_configs_l = 0
        for x in range(-l, l+1):
            for y in range(-l, l+1):
                if max(abs(x), abs(y)) != l:
                    continue
                num_configs_l += 1
        print(f"The link of length {l} has {num_configs_l} possible configurations")
        num_configs *= num_configs_l
    print(f"The total number of configuration for all 8 links iss {num_configs}")


def xy_path(grid_size: int = 128):
    path = [(0, 0)]

    x = 0
    y = 0
    while y < grid_size:
        y += 1
        path.append((x, y))

    # pt1
    assert x == 0
    assert y == grid_size
    print(f"Successfully reached pt1: ({x}, {y})\n{path}")

    while x < grid_size:
        x += 1
        path.append((x, y))

    # pt2
    assert x == grid_size
    assert y == grid_size
    print(f"Successfully reached pt2: ({x}, {y})\n{path}")

    x_direction = -1
    y_direction = -1
    y += y_direction
    while y >= 0:
        while 1 <= x <= grid_size:
            path.append((x, y))
            x += x_direction
        # Correct overshoot
        x -= x_direction
        # Change directions
        x_direction = -x_direction
        y += y_direction

    # pt3
    assert x == grid_size and y == -1, f"Expected pt3:({grid_size}, {0}), got ({x}, {y})\n{path}"
    print(f"Successfully reached pt5: ({x}, {y})\n{path}")

    x_direction = -1
    y_direction = -1
    while x >= 0:
        while -grid_size <= y <= -1:
            path.append((x, y))
            y += y_direction
        # Correct overshoot
        y -= y_direction
        # Change directions
        y_direction = -y_direction
        x += x_direction

    # pt6
    assert x == -1 and y == -grid_size, f"Expected pt6:({-1}, {-grid_size}), got ({x}, {y})\n{path}"
    print(f"Successfully reached pt6: ({x}, {y})\n{path}")

    x_direction = -1
    y_direction = 1
    while y <= 0:
        while -grid_size <= x <= -1:
            path.append((x, y))
            x += x_direction
        # Correct overshoot
        x -= x_direction
        x_direction = -x_direction
        y += y_direction

    # pt7
    assert x == -grid_size and y == 1, f"Expected pt7:({-grid_size}, {1}), got ({x}, {y})\n{path}"
    print(f"Successfully reached pt7: ({x}, {y})\n{path}")

    while x <= -1:
        while 1 <= y <= grid_size:
            path.append((x, y))
            y += y_direction
        # Correct overshoot
        y -= y_direction
        y_direction = -y_direction
        x += x_direction

    # pt8
    assert x == 0 and y == 1, f"Expected pt8:({0}, {1}), got ({x}, {y})\n{path}"
    print(f"Successfully reached pt8: ({x}, {y}) (not included in path)")

    assert len(path) == (2 * grid_size + 1) * (2 * grid_size + 1), f"Path has {len(path)} pts instead of {grid_size * grid_size}"
    assert len(set(path)) == (2 * grid_size + 1) * (2 * grid_size + 1), f"Path has {len(path)} unique pts instead of {(2 * grid_size + 1) * (2 * grid_size + 1)}"
    path.append((0, 0))

    # plt.plot([xy[0] for xy in path], [xy[1] for xy in path])
    # plt.show()

    return(path)





