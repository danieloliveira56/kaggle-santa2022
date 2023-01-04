from LKH import solve_lkh
from common import load_submission, IMAGE_LUT, image, evaluate_solution, cost_array, reference_config_cost, solution_to_submission, cost_map
import numpy as np

def solve():
    slice_idx = 27310
    base_solution = load_submission("submission79819.csv")
    config_pool = base_solution[:slice_idx]

    print(evaluate_solution(config_pool))
    print(evaluate_solution(np.concatenate((config_pool, config_pool[:1]))))

    lkh_solution = solve_lkh(config_pool=config_pool)

    print(lkh_solution.shape)
    print(evaluate_solution(lkh_solution))
    print(evaluate_solution(np.concatenate((lkh_solution, lkh_solution[:1]))))

    lkh_solution = np.concatenate((lkh_solution, base_solution[slice_idx:]))
    solution_to_submission(lkh_solution)

if __name__ == '__main__':
    solve()

