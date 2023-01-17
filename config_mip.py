import gurobipy as gp
import numpy as np
from gurobipy import GRB, tuplelist

from common import ARM_LENGTHS, config_to_string


def config_mip(
    xy_solution,
    links=8,
    UB=None,
    linear_relaxation=False,
    save_model=True,
    start_config=None,
    end_config=None,
):
    print("\nRunning MIP to find configs:")
    print(f"\txy_solution size: {len(xy_solution)}")

    print(f"links={links}")
    print(f"UB={UB}")
    print(f"linear_relaxation={linear_relaxation}")
    print(f"save_model={save_model}")
    print(f"start_config={config_to_string(start_config) if start_config is not None else ''}")
    print(f"start_config_xy={start_config.sum(axis=0) if start_config is not None else ''}")
    print(f"end_config={config_to_string(end_config) if end_config is not None else''}")
    print(f"end_config={end_config.sum(axis=0) if end_config is not None else''}")
    print(f"xy_solution[0]={xy_solution[0]}")

    print("Creating model...")
    m = gp.Model("configs")

    print(f"Creating {len(xy_solution)} arcs tuplelist")
    arcs = tuplelist(
        range(len(xy_solution)-1)
    )

    print(f"Creating config_arcs tuplelist")
    config_arcs = tuplelist(
        (i, j, x1, y1, x2, y2)
        for i in arcs
        for j in range(links)
        for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j] + 1)
        if max(abs(x1), abs(y1)) == ARM_LENGTHS[j]
        for x2 in range(max(x1-1, -ARM_LENGTHS[j]), min(x1+2, ARM_LENGTHS[j]+1))
        for y2 in range(max(y1-1, -ARM_LENGTHS[j]), min(y1+2, ARM_LENGTHS[j]+1))
        if max(abs(x2), abs(y2)) == ARM_LENGTHS[j]
    )

    # Create x variables
    print(f"Creating {len(config_arcs)} x arc variables...")

    x = m.addVars(config_arcs, vtype=GRB.BINARY, name="x")
    z = m.addVars(arcs, vtype=GRB.CONTINUOUS, name="z")

    print("Setting obj")
    m.setObjective(
        z.sum("*"),
        GRB.MINIMIZE,
    )

    print(f"Creating {len(arcs)} ConfigFrom_x constraints...")
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
            for i in arcs
        ),
        "ConfigFrom_x",
    )

    print(f"Creating {len(arcs)} ConfigFrom_y constraints...")
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
            for i in arcs
        ),
        "ConfigFrom_y",
    )

    print(f"Creating {len(arcs)} ConfigTo_x constraints...")
    m.addConstrs(
        (
            (
                gp.quicksum(
                    x2 * x.sum(i, j, "*", "*", x2, "*")
                    for j in range(links)
                    for x2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i+1][0]
            )
            for i in arcs
        ),
        "ConfigTo_x",
    )

    print(f"Creating {len(arcs)} ConfigTo_y constraints...")
    m.addConstrs(
        (
            (
                gp.quicksum(
                    y2 * x.sum(i, j, "*", "*", "*", y2)
                    for j in range(links)
                    for y2 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                )
                == xy_solution[i+1][1]
            )
            for i in arcs
        ),
        "ConfigTo_y",
    )

    print(f"Creating {len(arcs) * links} arcs constraints...")
    m.addConstrs(
        (
            x.sum(i, j, "*", "*", "*", "*") == 1
            for i in arcs
            for j in range(links)
        ),
        "SingleConfig",
    )

    print(f"Creating {(len(arcs)-1) * links} FlowConservation_x constraints...")
    m.addConstrs(
        (
            x.sum(i, j, "*", "*", x1, "*") == x.sum(i+1, j, x1, "*", "*", "*")
            for i in arcs[:-1]
            for j in range(links)
            for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
        ),
        "FlowConservation_x",
    )

    print(f"Creating {(len(arcs)-1) * links} FlowConservation_y constraints...")
    m.addConstrs(
        (
            x.sum(i, j, "*", "*", "*", y1) == x.sum(i+1, j, "*", y1, "*", "*")
            for i in arcs[:-1]
            for j in range(links)
            for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
        ),
        "FlowConservation_y",
    )

    if start_config is not None:
        print(f"Creating {links} Start0 constraints...")
        m.addConstrs(
            (
                x.sum(arcs[0], j, start_config[links-j-1][0], start_config[links-j-1][1], "*", "*") == 1
                for j in range(links)
            ),
            "Start0",
        )
    if end_config is not None:
        print(f"Creating {links} End0 constraints...")
        m.addConstrs(
            (
                x.sum(arcs[-1], j, "*", "*", end_config[links-j-1][0], end_config[links-j-1][1]) == 1
                for j in range(links)
            ),
            "End0",
        )

    print(f"Creating {len(arcs)} Costx constraints...")
    m.addConstrs(
        (
            (
                gp.quicksum(
                    x.sum(i, j, x1, "*", x2, "*")
                    for x1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    for x2 in range(max(x1-1, -ARM_LENGTHS[j]), min(x1+2, ARM_LENGTHS[j]+1))
                )
                <= z[i]
            )
            for i in arcs
            for j in range(links)
        ),
        "Costx",
    )

    print(f"Creating {len(arcs)} Costy constraints...")
    m.addConstrs(
        (
            (
                gp.quicksum(
                    x.sum(i, j, "*", y1, "*", y2)
                    for y1 in range(-ARM_LENGTHS[j], ARM_LENGTHS[j]+1)
                    for y2 in range(max(y1 - 1, -ARM_LENGTHS[j]), min(y1 + 2, ARM_LENGTHS[j] + 1))
                )
                <= z[i]
            )
            for i in arcs
            for j in range(links)
        ),
        "Costy",
    )

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
        m.computeIIS()
        m.write("model.ilp")

        return None

    print(f"Objective: {m.objVal:.2f}")

    solution = np.zeros(shape=(len(arcs)+1, links, 2), dtype=int)
    for a in config_arcs:
        if x[a].x > 0.5:
            print(f"x{a}={x[a].x}")
            i, j, x1, y1, x2, y2 = a
            solution[i, links-j-1, 0] = x1
            solution[i, links-j-1, 1] = y1
            solution[i+1, links-j-1, 0] = x2
            solution[i+1, links-j-1, 1] = y2

    for i in arcs:
        if z[i].x > 0.5:
            print(f"z{i}={z[i].x}")

    return solution
