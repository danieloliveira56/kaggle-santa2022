import numpy as np
from common import cost, is_valid, evaluate_solution, reference_config_cost, cost_map
import subprocess
import datetime
from numba import njit
from rich.progress import track
from rich.progress import Progress

# def two_opt():


def solve_lkh(config_pool, trace_level=1, precision=3, edge_data="EDGE_LIST", time_limit=60):

    start = datetime.datetime.now()

    print("\nRunning solve_atsp:")
    print(f"\tconfig_pool size: {len(config_pool)}")
    print(f"\tInitial cost: {evaluate_solution(config_pool)}")

    print("Writing LKH input files...")

    SIZE = len(config_pool)

    # WRITE PROBLEM FILE
    with open(f'group.par', 'w') as f:
        f.write("PROBLEM_FILE = distances.vrp\n")
        f.write("OUTPUT_TOUR_FILE = output.txt\n")
        f.write("INITIAL_TOUR_FILE = initial.txt\n")
        f.write("\n")

        # f.write("RUNS = 0\n")
        f.write("MOVE_TYPE = 5 SPECIAL\n")
        # f.write("GAIN23 = YES\n")
        f.write("KICKS = 1\n")
        f.write("KICK_TYPE = 4\n")
        f.write("POPULATION_SIZE = 100\n")
        f.write("MAX_SWAPS = 0\n")
        # f.write("MAX_TRIALS = 0\n")

        # f.write(f"BACKTRACKING = YES\n")
        # f.write(f"MOVE_TYPE = 2\n")
        # f.write(f"POPULATION_SIZE = 10\n")
        f.write(f"TIME_LIMIT = {time_limit}\n")  # seconds
        # f.write(f"POPMUSIC_INITIAL_TOUR = YES\n")  # seconds
        # f.write(f"INITIAL_PERIOD = 100\n")  # seconds
        f.write(f"TRACE_LEVEL = {trace_level}\n")  # seconds
        # f.write("MAX_TRIALS = 10000\n")  # seconds

    print("Creating cost_map")
    pool_cost_map = cost_map(config_pool)

    # WRITE PARAMETER FILE
    with open(f'distances.vrp', 'w') as f:
        f.write("NAME: distances\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {SIZE+1}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")

        if edge_data == "MATRIX":
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")

            cost_matrix = 100 * np.ones(shape=(SIZE, SIZE))
            for i in track(range(SIZE)):
                for j, ij_cost in pool_cost_map[i].items():
                    # if j <= i:
                    #     continue
                    cost_matrix[i][j] = ij_cost
                    cost_matrix[j][i] = ij_cost
            np.savetxt(f, cost_matrix, fmt=f"%.{precision}f")
        elif edge_data == "EDGE_LIST":
            f.write("EDGE_DATA_FORMAT: EDGE_LIST\n")
            f.write("EDGE_DATA_SECTION\n")
            f.write(f"1 2 0\n")
            f.write(f"1 {SIZE+1} 0\n")
            for i in track(range(SIZE)):
                # f.write(f"{i + 2} 1 0\n")
                for j, ij_cost in pool_cost_map[i].items():
                    if i >= j:
                        continue
                    f.write(f"{i+2} {j+2} {pow(10, precision) * ij_cost:.0f}\n")
        f.write("EOF\n")

    with open(f'initial.txt', 'w') as f:
        f.write("NAME: initial\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {SIZE+1}\n")
        f.write("TOUR_SECTION\n")
        for j in range(SIZE+1):
            f.write(f"{j+1}\n")

        f.write("-1\n")
        f.write("EOF\n")

    print("Finished writing LKH input files,", datetime.datetime.now() - start)

    # EXECUTE TSP SOLVER
    print("Running LKH")
    try:
        subprocess.run(
            [
                './LKH',
                'group.par',
            ],
        )
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    tour = []
    # READ RESULTING ORDER
    with open('output.txt') as f:
        line = ""
        while line != "TOUR_SECTION":
            line = f.readline().replace('\n', '')
        line = f.readline().replace('\n', '')
        while line != "-1":
            if line != "1":
                tour.append(int(line)-1)
            line = f.readline().replace('\n', '')

    print(tour)
    solution = [config_pool[i-1] for i in tour]

    return np.array(solution)