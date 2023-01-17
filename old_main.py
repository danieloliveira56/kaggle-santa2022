import csv
from functools import reduce
from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def solve():
    df_image = pd.read_csv("image.csv")
    image = df_to_image(df_image)

    plt.imshow(image, interpolation="nearest")
    plt.show()

    submission = []
    with open("sample_submission.csv", "r") as f:
        f.readline()
        line = f.readline()
        while line:
            position = []
            for vec in line.split(";"):
                x, y = vec.split(" ")
                position.append((int(x), int(y)))

            submission.append(position)
            line = f.readline()

    # print(image)
    score = evaluate(image, submission)
    print(score)


def cartesian_to_array(x, y, shape):
    m, n = shape[:2]
    i = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i, j


def array_to_cartesian(i, j, shape):
    m, n = shape[:2]
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    y = (n - 1) // 2 - i
    x = j - (n - 1) // 2
    return x, y


point = (1, 8)
shape = (9, 9, 3)
assert cartesian_to_array(*array_to_cartesian(*point, shape), shape) == point


# Functions to map an image between array and record formats
def image_to_dict(image):
    image = np.atleast_3d(image)
    kv_image = {}
    for i, j in np.product(range(len(image)), repeat=2):
        kv_image[array_to_cartesian(i, j, image.shape)] = tuple(image[i, j])
    return kv_image


def image_to_df(image):
    return pd.DataFrame(
        [(x, y, r, g, b) for (x, y), (r, g, b) in image_to_dict(image).items()],
        columns=["x", "y", "r", "g", "b"],
    )


def df_to_image(df_image):
    side = int(len(df_image) ** 0.5)  # assumes a square image
    return df_image.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)


# Cost of reconfiguring the robotic arm: the square root of the number of links rotated
def reconfiguration_cost(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    return np.sqrt(diffs.sum())


# Cost of moving from one color to another: the sum of the absolute change in color components
def color_cost(from_position, to_position, image, color_scale=3.0):
    return np.abs(image[to_position] - image[from_position]).sum() * color_scale


def get_position(config):
    return reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))


# Total cost of one step: the reconfiguration cost plus the color cost
def step_cost(from_config, to_config, image):
    from_position = cartesian_to_array(*get_position(from_config), image.shape)
    to_position = cartesian_to_array(*get_position(to_config), image.shape)
    return reconfiguration_cost(from_config, to_config) + color_cost(
        from_position, to_position, image
    )


def evaluate(image, submission):
    previous_position = submission[0]
    cost = 0

    for position in submission[1:]:
        cost += step_cost(previous_position, position, image)
        previous_position = position

    pts = [(x, y) for x in range(-127, 128) for y in range(-127, 128)]
    shape = (257, 257, 3)

    dist_matrix = [
        [
            step_cost(
                cartesian_to_array(x1, y1, shape),
                cartesian_to_array(x2, y2, shape),
                image,
            )
            for (x1, y1) in pts
        ]
        for (x2, y2) in pts
    ]

    with open("dist_matrix.txt", "w") as f:
        for i, row in enumerate(dist_matrix):
            for j, col in enumerate(row):
                f.write(f"{i} {j} {col}\n")

    print(cost)


if __name__ == "__main__":
    solve()
