import datetime
import subprocess

from rich.progress import track

from common import cost_xy, xy_path


def solve_lkh_xy(max_dist=2, precision=3):

    start = datetime.datetime.now()

    pts = xy_path(128)
    print(len(pts))
    pt_map = {
        xy: i
        for i, xy in enumerate(pts[:-1])
    }

    print("\nRunning solve_lkh xy...")

    SIZE = len(pts)

    initial_cost = 0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        initial_cost += cost_xy(x1, y1, x2, y2)
        assert abs(x1-x2) <= 1
        assert abs(y1-y2) <= 1
    print(initial_cost)

    print("Writing LKH input files...")

    # WRITE PROBLEM FILE
    with open(f"group.par", "w") as f:
        f.write("PROBLEM_FILE = distances.vrp\n")
        f.write("OUTPUT_TOUR_FILE = output.txt\n")
        f.write("INITIAL_TOUR_FILE = initial.txt\n")

        f.write("\n")

        f.write("MOVE_TYPE = 5 SPECIAL\n")
        f.write("KICKS = 1\n")
        f.write("KICK_TYPE = 4\n")
        f.write("POPULATION_SIZE = 100\n")
        f.write("MAX_SWAPS = 0\n")
        f.write("INITIAL_PERIOD = 100\n")

    with open(f"initial.txt", "w") as f:
        f.write("NAME: initial\n")
        f.write("TYPE: TOUR\n")
        f.write(f"DIMENSION: {SIZE+1}\n")
        f.write("TOUR_SECTION\n")
        for j in range(SIZE + 1):
            f.write(f"{j+1}\n")

        f.write("-1\n")
        f.write("EOF\n")

    # WRITE PARAMETER FILE
    with open(f"distances.vrp", "w") as f:
        f.write("NAME: distances\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {SIZE+1}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")

        f.write("EDGE_DATA_FORMAT: EDGE_LIST\n")
        f.write("EDGE_DATA_SECTION\n")
        f.write(f"1 2 0\n")
        f.write(f"1 {SIZE+1} 0\n")

        for xi, yi in pts:
            for delta_x in range(-max_dist, max_dist+1):
                for delta_y in range(-max_dist + abs(delta_x), max_dist + 1 - abs(delta_x)):
                    xj = xi + delta_x
                    if xj < -128 or xj > 128:
                        continue
                    yj = yi + delta_y
                    if yj < -128 or yj > 128:
                        continue
                    i = pt_map[(xi, yi)]
                    j = pt_map[(xj, yj)]
                    if i >= j:
                        continue

                    f.write(f"{i+2} {j+2} {pow(10, precision) * cost_xy(xi, yi, xj, yj):.0f}\n")
                    if i == 0:
                        f.write(f"{SIZE+1} {j + 2} {pow(10, precision) * cost_xy(xi, yi, xj, yj):.0f}\n")

        # for i in track(range(SIZE)):
        #     for j in range(SIZE):
        #         if i == j:
        #             continue
        #         x1, y1 = pts[i]
        #         x2, y2 = pts[j]
        #         if abs(x1-x2) + abs(y1-y2) > 4:
        #             continue
        #
        #         f.write(f"{i+2} {j+2} {pow(10, 3) * cost_xy(x1, y1, x2, y2):.0f}\n")
        f.write("EOF\n")

    print("Finished writing LKH input files,", datetime.datetime.now() - start)

    # EXECUTE TSP SOLVER
    print("Running LKH")
    try:
        cmd = [
            "./LKH",
            "group.par",
        ]
        subprocess.run(
            cmd, universal_newlines=True
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    tour = []
    # READ RESULTING ORDER
    with open("output.txt") as f:
        line = ""
        while line != "TOUR_SECTION":
            line = f.readline().replace("\n", "")
        line = f.readline().replace("\n", "")
        while line != "-1":
            if line != "1":
                tour.append(int(line) - 1)
            line = f.readline().replace("\n", "")

    print(tour)
    solution = [pts[i - 1] for i in tour]

    final_cost = 0
    for (x1, y1), (x2, y2) in zip(solution[:-1], solution[1:]):
        final_cost += cost_xy(x1, y1, x2, y2)
    print(final_cost)

    with open("xy_solution.csv", 'w') as f:
        f.write("x,y")
        for x, y in solution:
            f.write(f"{x},{y}")
