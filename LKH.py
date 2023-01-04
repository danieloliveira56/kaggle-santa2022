import numpy as np
from common import cost, is_valid, evaluate_solution, reference_config_cost, cost_map
import subprocess
import datetime
from numba import njit
from rich.progress import track

# def two_opt():


def solve_lkh(config_pool):

    start = datetime.datetime.now()

    print("\nRunning solve_atsp:")
    print(f"\tconfig_pool size: {len(config_pool)}")

    print("Writing LKH input files...")

    SIZE = len(config_pool)

    # WRITE PROBLEM FILE
    with open(f'group.par', 'w') as f:
        f.write("PROBLEM_FILE = distances.vrp\n")
        f.write("OUTPUT_TOUR_FILE = output.txt\n")
        f.write("INITIAL_TOUR_FILE = initial.txt\n")
        f.write("\n")

        # f.write("MOVE_TYPE = 5 SPECIAL\n")
        # f.write("GAIN23 = NO\n")
        # f.write("KICKS = 2\n")
        # f.write("MAX_SWAPS = 0\n")
        # f.write("POPULATION_SIZE = 100\n")

        f.write("RUNS = 1\n")
        f.write(f"TIME_LIMIT = {5 * 60 * 60}\n")  # seconds
        f.write("TRACE_LEVEL = 1\n")  # seconds
        # f.write("MAX_TRIALS = 10000\n")  # seconds
        f.write("RUNS = 10\n")  # seconds

    print("Creating cost_map")
    pool_cost_map = cost_map(config_pool)

    # WRITE PARAMETER FILE
    with open(f'distances.vrp', 'w') as f:
        f.write("NAME: distances\n")
        f.write("TYPE: ATSP\n")
        f.write(f"DIMENSION: {SIZE}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in track(range(SIZE), description="Writing distances.vrp..."):
            hash_i = np.array2string(config_pool[i])
            i_cost_map = pool_cost_map[hash_i]
            for j in range(i+1, SIZE):
                hash_j = np.array2string(config_pool[j])
                f.write(f"{int(1000 * i_cost_map.get(hash_j, 1000)): d}")

            # array_str = np.array2string(1_000 * reference_config_cost(config_pool[i], config_pool[i+1:]), precision=0, max_line_width=SIZE*20, threshold=SIZE*2)
            # f.write(array_str[1:-1])
            f.write("\n")

        f.write("EOF\n")

    with open(f'initial.txt', 'w') as f:
        f.write("NAME: initial\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {SIZE}\n")
        f.write("TOUR_SECTION\n")
        for j in range(SIZE):
            f.write(f"{j+1}\n")

        f.write("-1\n")
        f.write("EOF\n")

    print("Finished writing LKH input files,", datetime.datetime.now() - start)

    # EXECUTE TSP SOLVER\
    print("Running LKH")
    subprocess.run(
        [
            './LKH',
            'group.par',
        ],
    )

    tour = []
    # READ RESULTING ORDER
    with open('output.txt') as f:
        line = ""
        while line != "TOUR_SECTION":
            line = f.readline().replace('\n', '')
        line = f.readline().replace('\n', '')
        while line != "-1":
            tour.append(int(line))
            line = f.readline().replace('\n', '')

    print(tour)
    solution = [config_pool[i-1] for i in tour]
    solution.append(solution[0])

    return np.array(solution)