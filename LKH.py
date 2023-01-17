import datetime
import itertools
import os
import random
import re
import signal
import subprocess

import numpy as np
from rich.progress import track

from common import (cost_map, evaluate_solution, remove_replacements,
                    solution_to_submission, solution_to_xy_config_dict,
                    validate_solution)


def get_solution(config_pool, filename="output.txt"):
    tour = []
    # READ RESULTING ORDER
    with open(filename) as f:
        line = ""
        while line != "TOUR_SECTION":
            line = f.readline().replace("\n", "")
        line = f.readline().replace("\n", "")
        while line != "-1":
            if line != "1":
                tour.append(int(line) - 1)
            line = f.readline().replace("\n", "")

    return [config_pool[i - 1] for i in tour]


def solve_lkh(
    config_pool,
    trace_level=1,
    precision=3,
    edge_data="EDGE_LIST",
    time_limit=None,
    duplicate_xy=False,
    allow_replacements=False,
):

    start = datetime.datetime.now()

    initial_cost = evaluate_solution(config_pool)
    num_infeasible_edges = validate_solution(config_pool)[0]

    print("\nRunning solve_lkh...")
    print(f"\tconfig_pool size: {len(config_pool)}")
    print(f"\tInitial cost: {initial_cost}")
    if num_infeasible_edges > 0:
        print("Starting solution is valid")
    else:
        print("Starting solution is invalid")

    num_replacements = 0
    if allow_replacements:
        xy_config_map = solution_to_xy_config_dict(config_pool)

    if duplicate_xy:
        print("Duplicating xy alternatives...")
        xy_config_map = solution_to_xy_config_dict(config_pool)
        to_duplicate = []
        for _, configs in xy_config_map.items():
            if len(configs) > 1:
                to_duplicate.extend(configs)
        to_duplicate = sorted(to_duplicate, reverse=True)
        for i in to_duplicate:
            config_pool = np.concatenate([config_pool[: i + 1], config_pool[i:]])
        print(f"\tDuplicated config_pool size: {len(config_pool)}")
        print(f"\tDuplicated cost: {evaluate_solution(config_pool)}")

    print("Creating cost_map")
    pool_cost_map = cost_map(config_pool, duplicate_xy)

    SIZE = len(config_pool)

    print("Writing LKH input files...")
    try:
        os.remove("output.txt")
    except Exception:
        pass

    # WRITE PROBLEM FILE
    with open(f"group.par", "w") as f:
        f.write("PROBLEM_FILE = distances.vrp\n")
        f.write("OUTPUT_TOUR_FILE = output.txt\n")
        # f.write("INITIAL_TOUR_ALGORITHM = GREEDY\n")
        if not allow_replacements:
            f.write("INITIAL_TOUR_FILE = initial.txt\n")
        f.write("\n")

        # f.write("RUNS = 10\n")
        f.write("MOVE_TYPE = 5 SPECIAL\n")
        # f.write("PRECISION = 1\n")
        # f.write("GAIN23 = NO\n")
        f.write("KICKS = 1\n")
        f.write("KICK_TYPE = 4\n")
        # f.write("RESTRICTED_SEARCH = NO\n")
        f.write("POPULATION_SIZE = 100\n")
        f.write("MAX_SWAPS = 0\n")
        # f.write("MAX_CANDIDATES = 20\n")
        # f.write("CANDIDATE_SET_TYPE = POPMUSIC\n")
        # f.write("MAX_TRIALS = 0\n")
        # f.write(f"INITIAL_PERIOD = 1000\n")

        # f.write(f"BACKTRACKING = YES\n")
        # f.write(f"MOVE_TYPE = 2\n")
        # f.write(f"POPULATION_SIZE = 10\n")
        if time_limit:
            f.write(f"TIME_LIMIT = {time_limit}\n")  # seconds
        # f.write(f"POPMUSIC_INITIAL_TOUR = YES\n")  # seconds

        f.write(f"TRACE_LEVEL = {trace_level}\n")  # seconds
        # f.write("MAX_TRIALS = 10000\n")  # seconds

    # WRITE PARAMETER FILE
    with open(f"distances.vrp", "w") as f:
        f.write("NAME: distances\n")
        if allow_replacements:
            f.write("TYPE: ATSP\n")
        else:
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

            # Adjust cost_map arc costs
            if allow_replacements:
                for xy, configs in xy_config_map.items():
                    if xy == (0, 0):
                        continue

                    # Configurations at the same (x,y) point are be replaced, only one can be visited (set TSP)
                    cycle_configs = list(configs)
                    if len(configs) <= 1:  # or len(configs) > 2:
                        continue

                    # print(f"SET TSP Configs: {configs}")
                    # for i in configs:
                    #     print(config_pool[i])

                    # Save the cost map of config in cycle such as to turn them clockwise by 1.
                    cycle_cost_map = [pool_cost_map[i] for i in cycle_configs]
                    for i, cycle_config in enumerate(cycle_configs):
                        # Assign to a config the cost_map of the preceding config
                        pool_cost_map[cycle_config] = cycle_cost_map[i - 1]

                        # Adjust the cost map of the outside circle neighbor to avoid inconsistencies and asymetric costs
                        # Removed, only out arcs need to be changed
                        # for arc_cost, outside_config in pool_cost_map[
                        #     cycle_config
                        # ].items():
                        #     pool_cost_map[outside_config][cycle_config] = arc_cost

                    # Assign zero cost between cycle configurations
                    for i, j in itertools.combinations(configs, 2):
                        pool_cost_map[i][j] = 0
                        pool_cost_map[j][i] = 0

                    num_replacements += len(configs) - 1
                if num_replacements == 0:
                    print(f"Trying to replace {num_replacements} configs")
                    return None

                print(f"Trying to replace {num_replacements} configs")

            for i in track(range(SIZE)):
                for j, ij_cost in pool_cost_map[i].items():
                    if i == j:
                        continue
                    if not allow_replacements and i > j:
                        continue
                    f.write(f"{i+2} {j+2} {pow(10, precision) * ij_cost:.0f}\n")
        f.write("EOF\n")

    with open(f"initial.txt", "w") as f:
        f.write("NAME: initial\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {SIZE+1}\n")
        f.write("TOUR_SECTION\n")
        for j in range(SIZE + 1):
            f.write(f"{j+1}\n")

        f.write("-1\n")
        f.write("EOF\n")

    print("Finished writing LKH input files,", datetime.datetime.now() - start)

    # EXECUTE TSP SOLVER
    print("Running LKH")
    try:
        cmd = [
            "./LKH",
            "group.par",
        ]
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, universal_newlines=True, preexec_fn=os.setsid
        ) as p:
            found_lb = False
            best_cost = 10 * initial_cost
            line = p.stdout.readline()
            quit_process = False
            while line and not quit_process:
                if "Writing OUTPUT_TOUR_FILE" in line and best_cost < initial_cost:
                    solution = get_solution(config_pool)
                    solution = remove_replacements(solution)
                    if validate_solution(solution):
                        print("Saving submission to csv.")
                        solution_to_submission(solution)
                        if (
                            allow_replacements
                            and len(solution) - len(config_pool) <= num_replacements
                        ):
                            print("Desired number of replacements found!")
                            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    else:
                        print("Solution not valid.")

                m = re.match(r"Cost = (?P<cost>\d+\.?\d*).*", line)
                if m:
                    cost = float(m.groupdict()["cost"]) / pow(10, precision)
                    if cost < best_cost:
                        best_cost = cost

                    print(line.replace(cost, f"{cost:.1f}"))
                else:
                    print(line, end="")

                if not found_lb:
                    m = re.match(r"Lower bound = (?P<lb>\d+\.?\d*).*", line)
                    if m:
                        found_lb = True
                        lb = float(m.groupdict()["lb"]) / pow(10, precision)
                        if lb > initial_cost:
                            print(
                                f"LB={lb} is larger than initial solution cost={initial_cost}, quitting..."
                            )
                            quit_process = True

                line = p.stdout.readline()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    try:
        solution = np.array(get_solution(config_pool), dtype=int)
        if evaluate_solution(solution) > initial_cost and validate_solution(solution)[0] >= num_infeasible_edges:
            solution = None
    except Exception:
        solution = None

    return solution
