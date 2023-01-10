from LKH import solve_lkh
from common import load_submission, IMAGE_LUT, image, evaluate_solution, cost_array, reference_config_cost, solution_to_submission, cost_map, is_valid
import numpy as np

def solve():
    base_solution = load_submission("submission-78739-Jan09-2025.csv")
    initial_cost = evaluate_solution(base_solution)

    step_size = base_solution.shape[0]
    for start_idx in range(0, base_solution.shape[0], step_size):
        end_idx = min(start_idx + step_size, base_solution.shape[0])
        print("Searching:", start_idx, end_idx)
        config_pool = base_solution[start_idx:end_idx]

        print(config_pool.shape)

        lkh_solution = solve_lkh(config_pool=config_pool, trace_level=1, precision=3, time_limit=60*60)

        print("lkh_solution shape:", lkh_solution.shape)
        print(evaluate_solution(lkh_solution))

        lkh_solution = np.concatenate((base_solution[:start_idx], lkh_solution, base_solution[end_idx:]))
        for i in range(lkh_solution.shape[0] - 1, 1, -1):
            if np.abs(lkh_solution[i] - lkh_solution[i - 1]).sum() == 0:
                lkh_solution = np.concatenate([lkh_solution[:i], lkh_solution[i + 1:]])

        lkh_cost = evaluate_solution(lkh_solution)
        print(lkh_cost)
        if lkh_cost < initial_cost:
            print("saving better solution")
            base_solution = lkh_solution
            solution_to_submission(base_solution)


def unit_tests():
    config1 = np.array(
        [[64, 0],
        [32, 0],
        [16, 0],
        [8, 0],
        [4, 0],
        [2, 0],
        [1, 0],
        [1, 0]]
    )

    config2 = np.array(
        [[64, 1],
        [32, 1],
        [16, 1],
        [8, 1],
        [4, 1],
        [2, 1],
        [1, 1],
        [1, 1]]
    )

    assert is_valid(config1, config2)


if __name__ == '__main__':
    unit_tests()
    solve()