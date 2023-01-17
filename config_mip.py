import gurobipy as gp
import numpy as np
from gurobipy import GRB, tuplelist

from common import ARM_LENGTHS


def config_mip(
    xy_solution,
    links=8,
    UB=None,
    linear_relaxation=False,
    save_model=True,
    start=False,
    end=False,
):
    print("\nRunning MIP to find configs:")
    print(f"\txy_solution size: {len(xy_solution)}")

    print("Creating model...")
    m = gp.Model("configs")

    # Set variable type
    var_type = GRB.INTEGER
    if linear_relaxation:
        var_type = GRB.SEMICONT

    pts = tuplelist(
        range(len(xy_solution))
    )
    config_arcs = tuplelist(
        (j, x1, y1, x2, y2)
        for j in range(links)
        for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        if max(abs(x1), abs(y1)) == ARM_LENGTHS[j]
        for x2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        for y2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        if max(abs(x2), abs(y2)) == ARM_LENGTHS[j] and abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1
    )

    # arcs = tuplelist(
    #     (i, link_length, x1, y1, x2, y2)
    #     for i in range(len(xy_solution))
    #     for x1 in range(-link_length, link_length+1)
    #     for y1 in range(-link_length, link_length+1)
    #     if max(abs(x1), abs(y1)) == link_length
    #     for x2 in range(-link_length, link_length+1)
    #     for y2 in range(-link_length, link_length+1)
    #     if max(abs(x2), abs(y2)) == link_length and abs(x2-x1) <= 1 and abs(y2-y1) <= 1
    #     for link_length in [1, 1, 2, 4, 8, 16, 32, 64]
    # )
    arcs = tuplelist(
        (i, config_arc)
        for i in pts
        for config_arc in config_arcs
    )
    arcs = tuplelist(
        (i, j, x1, y1, x2, y2)
        for i in pts
        for j in range(links)
        for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        if max(abs(x1), abs(y1)) == ARM_LENGTHS[j]
        for x2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        for y2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        if max(abs(x2), abs(y2)) == ARM_LENGTHS[j] and abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1
    )

    arc_costs = tuplelist(
        i for i in range(len(xy_solution))
    )
    print(pts)

    # Create x variables
    print(f"\nCreating {len(arcs)} x arc variables...")

    x = m.addVars(arcs, vtype=var_type, lb=0, ub=1, name="x")
    z = m.addVars(arc_costs, vtype=GRB.SEMICONT, name="z")

    print("Setting obj\n")
    m.setObjective(
        z.sum("*"),
        GRB.MINIMIZE,
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    x1 * x.sum(i, j, x1, "*", "*", "*")
                    for j in range(links)
                    for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i][0]
            )
            for i in pts[:-1]
        ),
        "ConfigFrom_x",
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    y1 * x.sum(i, j, "*", y1, "*", "*")
                    for j in range(links)
                    for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i][1]
            )
            for i in pts[:-1]
        ),
        "ConfigFrom_y",
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    x2 * x.sum(i-1, j, "*", "*", x2, "*")
                    for j in range(links)
                    for x2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i][0]
            )
            for i in pts[1:]
        ),
        "ConfigTo_x",
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    y2 * x.sum(i-1, j, "*", "*", "*", y2)
                    for j in range(links)
                    for y2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i][1]
            )
            for i in pts[1:]
        ),
        "ConfigTo_y",
    )

    m.addConstrs(
        (
            x.sum(i, j, "*", "*", "*", "*") == 1
            for i in pts
            for j in range(links)
        ),
        "SingleConfig",
    )

    m.addConstrs(
        (
            x.sum(i, j, "*", "*", x1, y1) == x.sum(i+1, j, x1, y1, "*", "*")
            for i in pts[:-1]
            for j in range(links)
            for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
            for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
        ),
        "FlowConservation",
    )

    if start:
        m.addConstrs(
            (
                x.sum(pts[0], j, ARM_LENGTHS[j] if j == links - 1 else -ARM_LENGTHS[j], 0, "*", "*") == 1
                for j in range(links)
            ),
            "Start0",
        )

    if end:
        m.addConstrs(
            (
                x.sum(pts[-1], j, "*", "*", ARM_LENGTHS[j] if j == links - 1 else -ARM_LENGTHS[j], 0) == 1
                for j in range(links)
            ),
            "End0",
        )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    x.sum(i, j, x1, "*", x2, "*")
                    for j in range(links)
                    for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    for x2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    if abs(x1-x2) == 1
                )
                <= z[i]
            )
            for i in pts
        ),
        "Costx",
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    x.sum(i, j, "*", y1, "*", y2)
                    for j in range(links)
                    for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    for y2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    if abs(y1 - y2) == 1
                )
                <= z[i]
            )
            for i in pts
        ),
        "Costy",
    )

    # m.computeIIS()
    if save_model:
        print("Saving model...")
        m.write("santa_configs.lp")

    seen_tours = set()
    m._seen_tours = seen_tours

    # Model Parameters
    # m.setParam(GRB.Param.Heuristics, 0.001)
    # m.setParam(GRB.Param.Aggregate, 0)

    # m.setParam(GRB.Param.NoRelHeurTime, 36_000)
    # m.setParam(GRB.Param.Symmetry, 2)
    # m.setParam(GRB.Param.PreDepRow, 1)
    # m.setParam(GRB.Param., 1)
    m.setParam(GRB.Param.MIPFocus, 1)
    if UB:
        m.setParam(GRB.Param.Cutoff, UB)

    # m.tune()
    # if m.tuneResultCount > 0:
    #     # Load the best tuned parameters into the model
    #     m.getTuneResult(0)
    #     # Write tuned parameters to a file
    #     m.write('tune.prm')

    print("Optimizing...")
    m.optimize()

    print(m.Status)
    if m.Status == GRB.INFEASIBLE or m.Status == GRB.CUTOFF:
        return None

    print(f"Objective: {m.objVal:.2f}")

    solution = np.empty(shape=(len(pts), links, 2), dtype=int)
    for a in arcs:
        if x[a].x > 0.5:
            print(f"x{a}={x[a].x}")
            i, j, x1, y1, x2, y2 = a
            solution[i, links-j-1, 0] = x1
            solution[i, links-j-1, 1] = y1

    print(solution)

    for i in pts:
        if z[i].x > 0.5:
            print(f"z{i}={z[i].x}")

    return solution
