import numpy as np
import itertools
import rich.progress

from common import (BOTTOM_LEFT, BOTTOM_RIGHT, INITIAL_CONFIG, TOP_LEFT,
                    TOP_RIGHT, ARM_LENGTHS, compress_solution, cost, cost_map,
                    delete_indices, enumerate_quadrants, evaluate_solution,
                    get_axis_configs, get_neighbors, get_quadrant_path,
                    get_quadrant_solution, image, is_valid, load_submission,
                    plot_traj, print_solution, reference_config_cost,
                    remove_replacements, solution_to_submission,
                    solution_to_xy_config_dict, xy_path,
                    validate_solution, valid_config)
from LKH import solve_lkh
from LKH_xy import solve_lkh_xy
from config_mip import config_mip

NP_OFFSETS = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1]),
]
def solve_quadrant():
    solution = np.concatenate(
        [
            np.array([INITIAL_CONFIG]),
            get_quadrant_solution(TOP_LEFT),
            np.array([INITIAL_CONFIG]),
            get_quadrant_solution(TOP_RIGHT),
            np.array([INITIAL_CONFIG]),
            get_quadrant_solution(BOTTOM_RIGHT),
            np.array([INITIAL_CONFIG]),
            get_quadrant_solution(BOTTOM_LEFT),
            np.array([INITIAL_CONFIG]),
            get_axis_configs(),
            np.array([INITIAL_CONFIG]),
        ]
    )
    solution = solve_lkh(solution, duplicate_xy=True)
    print(solution.shape)
    validate_solution(solution)
    print(evaluate_solution(solution))
    solution_to_submission(solution)


def scalar_to_base(x, base):
    x_base = []
    x -= np.sum(base)
    for cur_x, l in zip(reversed(base), [1, 1, 2, 4, 8, 16, 32, 64]):
        if cur_x:
            x_base.append(cur_x)
            continue

        if x >= 0:
            x_base.append(min(x, l) - cur_x)
        else:
            x_base.append(-min(-x, l) + cur_x)
        x -= x_base[-1]

    if x:
        return None
    return np.array(list(reversed(x_base)), dtype=int)



def scalar_to_all_similar_configs(x, config):
    # Return all ways to distribute x among the non basic positions of base (which doesn't need to be a base per se)
    free_positions = [i for i, val in enumerate(config) if abs(val) != ARM_LENGTHS[i]]
    base = np.copy(config)
    for i in free_positions:
        base[i] = 0
    x -= np.sum(base)
    configs = np.empty(shape=(0, 8), dtype=int)
    for allocation in itertools.product(*[
        list(range(-ARM_LENGTHS[i], ARM_LENGTHS[i]+1))
        for i in free_positions
    ]):
    # for positions in itertools.combinations_with_replacement(list(range(x+1)), len(free_positions)-1):
        if sum(allocation) != x:
            continue
        new_config = np.copy(base)
        # i = 0
        # x_to_allocate = x
        # previous_pos = 0
        # while i < len(positions):
        #     cur_x = positions[i] - previous_pos
        #     previous_pos = i
        #     new_config[free_positions[i]] = cur_x
        #     i += 1
        for i, val in enumerate(allocation):
            new_config[free_positions[i]] = val
        if np.any(np.abs(new_config) > ARM_LENGTHS):
            continue
        configs = np.concatenate(
            [
                configs,
                np.array([new_config]),
            ]
        )

    return configs


def get_canonical_solution(canonical_base, x, y):
    # A canonical solution should maximize compatibility with neighboring solutions
    # Compatibility is maximized by having more links free to move
    x_config = scalar_to_base(x, canonical_base[:, 0])
    if x_config is None:
        return None
    y_config = scalar_to_base(y, canonical_base[:, 1])
    if y_config is None:
        return None

    return np.dstack([x_config, y_config])


def get_basic_neighborhood(config, x, y):
    # Return all configs at (x, y) with the same base as config.
    x_configs = scalar_to_all_similar_configs(x, config[:, 0])
    if x_configs.shape[0] == 0:
        return np.empty(shape=(0, 8, 2), dtype=int)
    y_configs = scalar_to_all_similar_configs(y, config[:, 1])
    if y_configs.shape[0] == 0:
        return np.empty(shape=(0, 8, 2), dtype=int)

    return np.concatenate([
        np.dstack([
            config_x,
            config_y
        ])
        for config_x, config_y in itertools.product(x_configs, y_configs)
    ])


def get_canonical_base(config):
    # A canonical solution should maximize compatibility with neighboring solutions
    # Compatibility is maximized by having more links free to move
    canonical_base = []
    for i, l in enumerate([64, 32, 16, 8, 4, 2, 1, 1]):
        if abs(config[i][0]) == l:
            canonical_base.append(np.array([config[i][0], 0], dtype=int))
        else:
            canonical_base.append(np.array([0, config[i][1]], dtype=int))

    return np.array(canonical_base, dtype=int)


def xy_neighborhood(x, y, max_dist=8):
    # Return all pts at most max_dist of x,y (Manhatan distance)
    for delta_x in range(-max_dist, max_dist + 1):
        if delta_x == 0:
            continue
        x2 = x + delta_x
        if x2 < -128 or x2 > 128:
            continue
        for delta_y in range(-max_dist + abs(delta_x), max_dist - abs(delta_x)):
            if delta_y == 0:
                continue
            y2 = y + delta_y
            if y < -128 or y > 128:
                continue
            yield x2, y2


def get_candidates_from_missing_pts(solution):
    print(f"Searching missing points to patch, initial size: {solution.shape}")

    xy_config_dict = solution_to_xy_config_dict(solution)
    missing_pts = [
        (x, y)
        for x in range(-128, 129)
        for y in range(-128, 129)
        if len(xy_config_dict[(x, y)]) == 0
    ]
    print(f"Found {len(missing_pts)} missing pts to patch")
    if len(missing_pts) == 0:
        return None

    print(missing_pts)

    candidates = np.empty(shape=(0, 8, 2), dtype=int)
    for x, y in missing_pts:
        candidates = np.unique(np.concatenate([
            candidates,
            find_cross_candidates(solution, x, y, max_dist=4)
        ]), axis=0)
    return candidates


def search_patches(solution):
    print(f"Searching missing points to patch, initial size: {solution.shape}")

    xy_config_dict = solution_to_xy_config_dict(solution)
    missing_pts = [
        (x, y)
        for x in range(-128, 129)
        for y in range(-128, 129)
        if len(xy_config_dict[(x, y)]) == 0
    ]
    print(f"Found {len(missing_pts)} missing pts to patch")
    print(missing_pts)

    neighbor_pts = set()
    for xi, yi in missing_pts:
        for delta_x in range(-8, 9):
            if delta_x == 0:
                continue
            for delta_y in range(-8 + abs(delta_x), 9 - abs(delta_x)):
                if delta_y == 0:
                    continue
                xj = xi + delta_x
                if xj < -128 or xj > 128:
                    continue
                yj = yi + delta_y
                if yj < -128 or yj > 128:
                    continue
                neighbor_pts.add((xj, yj))

    print(f"Surveying {len(neighbor_pts)} neighbor pts")

    canonical_bases = np.empty(shape=(0, 8, 2), dtype=int)
    for x, y in neighbor_pts:
        for config in xy_config_dict[(x, y)]:
            canonical_bases = np.concatenate(
                [
                    canonical_bases,
                    np.array([get_canonical_base(solution[config])], dtype=int),
                ]
            )

    print(f"Built {canonical_bases.shape} canonical base")
    canonical_bases = np.unique(canonical_bases, axis=0)
    print(f"Unique: {canonical_bases.shape} canonical base")

    candidate_pool = np.empty(shape=(0, 8, 2), dtype=int)
    for x, y in missing_pts:
        for base in canonical_bases:
            config = get_canonical_solution(base, x, y)
            if config is None:
                continue
            candidate_pool = np.concatenate([candidate_pool, config])

    print(
        f"Finished searching missing points to patch, candidate_pool size: {candidate_pool.shape}"
    )
    return candidate_pool


def config_to_base(config):
    base = np.copy(config)
    for i, l in enumerate(ARM_LENGTHS):
        if abs(base[i][0]) < l:
            base[i][0] = 0
        if abs(base[i][1]) < l:
            base[i][1] = 0
    return base


def find_cross_candidates(solution, x, y, max_dist):
    # Return candidates to fill coordinates (x,y) based on basic neighbors of solution
    xy_config_dict = solution_to_xy_config_dict(solution)

    xy_neighbors = np.array([
        solution[config]
        for xj, yj in xy_neighborhood(x, y, max_dist=max_dist)
        for config in xy_config_dict[xj, yj]
    ])

    candidates = np.empty(shape=(0, 8, 2), dtype=int)
    for config in xy_neighbors:
        xj, yj = config.sum(axis=0)
        x_delta = xj - x
        y_delta = yj - y
        if abs(x_delta) > max_dist:
            continue
        if abs(y_delta) > max_dist:
            continue
        for x_offset in NP_OFFSETS:
            for y_offset in NP_OFFSETS:
                new_config = config - np.dstack([x_delta * x_offset, y_delta * y_offset])
                if valid_config(new_config):
                    candidates = np.unique(np.concatenate([
                        candidates,
                        new_config.reshape(1, 8, 2)
                    ]), axis=0)

    return candidates

def find_candidates(solution, x, y, max_dist=8):
    # Return candidates to fill coordinates (x,y) based on basic neighbors of solution
    xy_config_dict = solution_to_xy_config_dict(solution)

    xy_neighbors_bases = np.array([
        config_to_base(solution[config])
        for xj, yj in xy_neighborhood(x, y, max_dist=max_dist)
        for config in xy_config_dict[xj, yj]
    ])
    xy_neighbors_bases = np.unique(xy_neighbors_bases, axis=0)

    candidates = np.empty(shape=(0, 8, 2), dtype=int)
    for config in xy_neighbors_bases:
        candidates = np.unique(np.concatenate([
            candidates,
            get_basic_neighborhood(config, x, y)
        ]), axis=0)

    return candidates


def greedy_patch(solution, candidate_pool):
    solution_xy_config_dict = solution_to_xy_config_dict(solution)
    candidates_pool_xy_config_dict = solution_to_xy_config_dict(candidate_pool)

    missing_pts = [
        (x, y)
        for x in range(-128, 129)
        for y in range(-128, 129)
        if len(solution_xy_config_dict[(x, y)]) == 0
    ]
    avg_costs = {(x, y): [] for (x, y) in missing_pts}
    min_costs = {(x, y): [] for (x, y) in missing_pts}
    degree = {(x, y): [] for (x, y) in missing_pts}
    # for i, candidate in enumerate(candidate_pool):
    #     if tuple(candidate.sum(axis=0)) not in missing_pts:
    #         continue
    #     candidate_costs = [
    #         cost(candidate, solution[config])
    #         for x, y in xy_neighborhood(*candidate.sum(axis=0))
    #         for config in solution_xy_config_dict[(x, y)]
    #         if is_valid(candidate, solution[config])
    #     ]
    #
    #     if candidate_costs:
    #         avg_costs[tuple(candidate.sum(axis=0))].append(
    #             (i, np.average(candidate_costs))
    #         )
    #         min_costs[tuple(candidate.sum(axis=0))].append((i, np.min(candidate_costs)))
    #         degree[tuple(candidate.sum(axis=0))].append((i, len(candidate_costs)))
    #
    # for pt in missing_pts:
    #     min_costs[pt] = sorted(min_costs[pt], key=lambda v: v[1])
    #     avg_costs[pt] = sorted(avg_costs[pt], key=lambda v: v[1])
    #     degree[pt] = sorted(degree[pt], key=lambda v: v[1])

    print("Starting greedy insertion search...")
    print(f"Starting size: {solution.shape}")
    print(f"Initial cost: {evaluate_solution(solution)}")

    print(f"Attempting patch")
    new_solution = np.copy(solution)
    for pt in missing_pts:
        if not candidates_pool_xy_config_dict[pt]:
            print(f"Point {pt} has no candidate configuration")
            continue
        best_i = None
        best_candidate = None
        best_cost = 50
        for candidate_idx in candidates_pool_xy_config_dict[pt]:
            candidate = candidate_pool[candidate_idx]
            for i in range(1, new_solution.shape[0]):
                if not is_valid(candidate, new_solution[i-1]) or not is_valid(candidate, new_solution[i]):
                    continue
                insertion_cost = cost(candidate, new_solution[i-1]) + cost(candidate, new_solution[i])
                if insertion_cost < best_cost:
                    best_i = i
                    best_cost = insertion_cost
                    best_candidate = candidate
        if best_i:
            print(f"Inserting {pt} at {best_i} with a cost of {best_cost}")
            print(best_candidate)
            new_solution = np.concatenate([
                new_solution[:best_i],
                np.copy(candidate).reshape((1, 8, 2)),
                new_solution[best_i:]
            ])
            print(f"Current size: {new_solution.shape}")
            print(f"Current cost: {evaluate_solution(new_solution)}")

    solution_to_submission(new_solution)
    return new_solution

def delete_invalid_pts(solution):
    for i in range(solution.shape[0] - 2, 1, -1):
        if not is_valid(solution[i], solution[i + 1]) and i < 4000:
            print(f"Deleting pt {i}")
            solution = np.concatenate(
                [solution[:i], solution[i + 2:]]
            )
            break
    return solution

def lkh_local_search(
    submission_file="submission.csv",
    step_size=None,
    start_idx=None,
    end_idx=None,
    allow_replacements=False,
    try_patching=False
):
    base_solution = load_submission(submission_file)
    initial_cost = evaluate_solution(base_solution)

    candidate_pool = None
    if try_patching:
        # candidate_pool = search_patches(base_solution)
        base_solution = delete_invalid_pts(base_solution)
        candidate_pool = get_candidates_from_missing_pts(base_solution)

        if candidate_pool is not None:
            base_solution = greedy_patch(base_solution, candidate_pool)

    search_ranges = []
    if step_size:
        search_ranges = [
            [i, min(i + step_size, base_solution.shape[0])]
            for i in range(0, base_solution.shape[0], step_size)
        ]

    if start_idx and end_idx:
        search_ranges = [[start_idx, end_idx]]

    if not search_ranges:
        search_ranges = [[0, base_solution.shape[0]]]

    for start_idx, end_idx in search_ranges:
        print("Searching:", start_idx, end_idx)

        if candidate_pool is not None:
            config_pool = np.concatenate([
                base_solution[start_idx:start_idx+1],
                candidate_pool,
                base_solution[start_idx+1:end_idx],
            ])
        else:
            config_pool = base_solution[start_idx:end_idx]

        print(config_pool.shape)

        lkh_solution = solve_lkh(
            config_pool=config_pool,
            trace_level=1,
            precision=3,
            allow_replacements=allow_replacements,
        )
        if lkh_solution is None:
            continue

        print("lkh_solution shape:", lkh_solution.shape)
        print(evaluate_solution(lkh_solution))

        candidate_solution = np.concatenate(
            (base_solution[:start_idx], lkh_solution, base_solution[end_idx:])
        )
        candidate_solution = remove_replacements(candidate_solution)

        # Remove duplicates
        for i in range(candidate_solution.shape[0] - 1, 1, -1):
            if np.abs(candidate_solution[i] - candidate_solution[i - 1]).sum() == 0:
                candidate_solution = np.concatenate(
                    [candidate_solution[:i], candidate_solution[i + 1 :]]
                )

        lkh_cost = evaluate_solution(candidate_solution)
        valid_solution = validate_solution(candidate_solution)
        if valid_solution:
            lkh_solution = candidate_solution
        print(f"New solution cost: {lkh_cost}")
        if lkh_cost < initial_cost:
            if valid_solution:
                print("Saving better solution")
                base_solution = lkh_solution
                solution_to_submission(base_solution)
                initial_cost = lkh_cost
            else:
                print("Saving invalid better solution")
                solution_to_submission(lkh_solution, save_as="invalid")


def plot_solution():
    solution = load_submission("78700submission.csv")
    plot_traj(solution, image)


def solve():

    # path = [(x,0) for x in range(1,129)] + [(128,y) for y in range(1,129)]
    #
    # x = 127
    # y = 128
    # y_direction = -1
    # while x > 0:
    #     while y >= 1 and y <= 128:
    #         path.append((x, y))
    #         y += y_direction
    #     y_direction = -y_direction
    #     x -= 1
    #     y += y_direction
    # a = get_quadrant_solution(TOP_RIGHT)
    # print(evaluate_solution(a[:2]))
    # print(evaluate_solution(a[-2:]))

    # top_right_solution = solve_lkh(get_quadrant_solution(TOP_RIGHT), duplicate_xy=False)
    # top_left_solution = solve_lkh(get_quadrant_solution(TOP_LEFT), duplicate_xy=False)
    # bottom_right_solution = solve_lkh(get_quadrant_solution(BOTTOM_RIGHT), duplicate_xy=False)
    # bottom_left_solution = solve_lkh(get_quadrant_solution(BOTTOM_LEFT), duplicate_xy=False)

    # solution = np.concatenate([
    #     top_right_solution[:-1],
    #     top_left_solution[1:-1],
    #     bottom_right_solution[1:-1],
    #     bottom_left_solution[1:],
    # ])

    # pts_seen = set()
    # to_delete = []
    # for i, config in enumerate(base_solution):
    #     x, y = config.sum(axis=0)
    #     if i < 1000 and (x, y) in pts_seen and x == -128:
    #         to_delete.append(i)
    #     pts_seen.add((x, y))
    # base_solution = delete_indices(base_solution, to_delete)

    # pts_slice = set()
    # step_size = 25000
    # for config in base_solution[:step_size]:
    #     pts_slice.add(tuple(config.sum(axis=0)))
    #
    # for x1 in [-64, -63]:
    #     for i in range(-32, 33):
    #         new_config = np.array([
    #             [[x1, -64],
    #              [-32, i],
    #              [-16, -16],
    #              [-8, -8],
    #              [-4, -4],
    #              [-2, -2],
    #              [-1, -1],
    #              [-1, -1]]
    #         ])
    #         if tuple(new_config[0].sum(axis=0)) in pts_slice:
    #             base_solution = np.concatenate([base_solution[:1], new_config, base_solution[1:]])
    #             step_size += 1
    #     for i in range(-16, 17):
    #         new_config = np.array([
    #             [[x1, -64],
    #              [-32, 32],
    #              [-16, i],
    #              [-8, -8],
    #              [-4, -4],
    #              [-2, -2],
    #              [-1, -1],
    #              [-1, -1]]
    #         ])
    #         if tuple(new_config[0].sum(axis=0)) in pts_slice:
    #             base_solution = np.concatenate([base_solution[:1], new_config, base_solution[1:]])
    #             step_size += 1
    #     for i in range(-8, -5):
    #         new_config = np.array([
    #             [[x1, -64],
    #              [-32, 32],
    #              [-16, 16],
    #              [-8, i],
    #              [-4, -4],
    #              [-2, -2],
    #              [-1, -1],
    #              [-1, -1]]
    #         ])
    #         print(new_config[0].sum(axis=0))
    #         if tuple(new_config[0].sum(axis=0)) in pts_slice:
    #             base_solution = np.concatenate([base_solution[:1], new_config, base_solution[1:]])
    #             step_size += 1
    #
    # for i in range(-16, 17):
    #     new_config = np.array([
    #         [[-64, -64],
    #          [-32, 32],
    #          [-16, i],
    #          [-8, -8],
    #          [-4, -4],
    #          [-2, -2],
    #          [-1, -1],
    #          [-1, -1]]
    #     ])
    #     if tuple(config.sum(axis=0)) in pts_slice:
    #         base_solution = np.concatenate([base_solution[:1], new_config, base_solution[1:]])
    #         step_size += 1

    print(f"Slice cost: {evaluate_solution(base_solution[:step_size])}")

    # Enrich neighbors
    enrich_neighbors = False
    if enrich_neighbors:
        slice = base_solution[:step_size]
        for i, config in rich.progress.track(enumerate(slice)):
            for n in get_neighbors(config, max_dist=1):
                if tuple(n.sum(axis=0)) in pts_slice:
                    base_solution = np.concatenate(
                        [base_solution[:1], np.array([n]), base_solution[1:]]
                    )
                    step_size += 1


def unit_tests():
    config1 = np.array(
        [[64, 0], [32, 0], [16, 0], [8, 0], [4, 0], [2, 0], [1, 0], [1, 0]]
    )

    config2 = np.array(
        [[64, 1], [32, 1], [16, 1], [8, 1], [4, 1], [2, 1], [1, 1], [1, 1]]
    )

    assert is_valid(config1, config2)


if __name__ == "__main__":
    links = 8
    max_mip_pts = 50
    n = [1, 2, 4, 8, 16, 32, 64, 128][links-1]
    # solution = load_submission("submission-2599-101-(0, 65948)-Jan17-0030.csv")
    # solution = np.empty(shape=(0, links, 2), dtype=int)
    solution = np.array([INITIAL_CONFIG], dtype=int)

    solution_xy = xy_path(n)
    solution_xy = solve_lkh_xy(solution_xy, n=n, max_dist=4)
    for i, xy in enumerate(solution_xy):
        print(f"{i}: {xy}")

    solution_pieces = []
    start = len(solution) - 1
    while start < len(solution_xy) - 1:
        print(i, start, len(solution_xy), solution_xy[i])
        i = start + 1
        while solution_xy[i] not in [(-n, -n), (-n, n), (n, -n), (n, n)] and i - start < max_mip_pts and i < len(solution_xy) - 1:
            i += 1
        solution_pieces.append(solution_xy[start:i+1])
        start = i

    assert np.all(solution_pieces[0][0] == solution[-1].sum(axis=0)), f"solution_xy[0][0]:{solution_pieces[0][0]}\nsolution[-1].sum(axis=0):{solution[-1].sum(axis=0)}"

    for j, _ in enumerate(solution_pieces):
        print(f"MIP Solving solution piece {j+1} of {len(solution_pieces)}")
        print(f"{solution_pieces[j]}")

        piece_solution = config_mip(
                solution_pieces[j],
                save_model=False,
                links=links,
                start_config=solution[-1],
                end_config=INITIAL_CONFIG if (j == len(solution_pieces) - 1) and solution_pieces[j][-1] == (0, 0) else None,
            )

        solution = np.concatenate([
            solution,
            piece_solution[1:],
        ])
        print(f"Current solution size: {solution.shape}")
        solution_to_submission(solution)

    print(solution)
    solution_to_submission(solution)

    exit()

    unit_tests()
    solve_lkh_xy()
    # solve()
    lkh_local_search(
        submission_file="submission-777378-67031-(0, 0)-Jan16-0557.csv",
        step_size=1000,
        # allow_replacements=True,
        # try_patching=True,
        # start_idx=3000,
        # end_idx=4000,
        allow_replacements=True
    )
    # lkh_local_search(submission_file="submission-780148-67169-Jan11-1416.csv")
    # get_quadrant_path()
