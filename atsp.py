import itertools
from itertools import combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB, tuplelist

from common import cost, is_valid


def solve_atsp(
    config_pool,
    initial_solution=None,
    UB=None,
    linear_relaxation=False,
    save_model=False,
    arc_limit=6,
    min_max_obj=True,
    dfj_model=True,
    mtz_model=False,
):

    pts = np.unique([np.sum(config, axis=0) for config in config_pool], axis=0)
    print(pts)
    print("\nRunning solve_atsp:")
    print(f"\tconfig_pool size: {len(config_pool)}")

    # config_pool = [tuple(tuple(arm) for arm in config) for config in config_pool]
    # for c in config_pool:
    #     print(c)

    print("Creating model...")
    m = gp.Model("santa2022")

    # Set variable type
    var_type = GRB.BINARY
    if linear_relaxation:
        var_type = GRB.SEMICONT

    arcs = tuplelist(
        (i, j)
        for (i, j) in itertools.product(range(len(config_pool)), repeat=2)
        if i != j and is_valid(config_pool[i], config_pool[j])
    )
    print(arcs)
    # Create x variables
    print(f"\nCreating {len(arcs)} x arc variables...")

    x = m.addVars(arcs, vtype=var_type, lb=0, ub=1, name="x")
    if mtz_model:
        u = m.addVars(config_pool, lb=0, vtype=GRB.INTEGER, name="u")
        if UB:
            Q = UB
        else:
            Q = 2429

    for i, j in arcs:
        print(i, j)

    print("Setting obj\n")
    m.setObjective(
        sum(x[i, j] * cost(config_pool[i], config_pool[j]) for (i, j) in arcs),
        GRB.MINIMIZE,
    )

    m.addConstrs(
        (
            (
                gp.quicksum(
                    x.sum("*", i)
                    for i in range(len(config_pool))
                    if tuple(np.sum(config_pool[i], axis=0)) == (pt_x, pt_y)
                )
                >= 1
            )
            for (pt_x, pt_y) in pts
        ),
        "ConfigTo",
    )

    print(f"Creating FlowConservation constraints...")
    m.addConstrs(
        ((x.sum("*", j) == x.sum(j, "*")) for j in range(len(config_pool))),
        "FlowConservation",
    )

    # m.computeIIS()
    if save_model:
        print("Saving model...")
        m.write("santa_atsp.lp")

    seen_tours = set()
    m._seen_tours = seen_tours

    def subtour(edges, tour_configs, start=None):
        is_subtour = False
        if start:
            tour_configs = [start, start] + tour_configs
        unvisited = [w for w in tour_configs]
        cycle = [w for w in tour_configs]  # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]
            if len(thiscycle) <= len(cycle):
                is_subtour = True
                cycle = thiscycle  # New shortest subtour
                return cycle, is_subtour
            else:
                cycle = thiscycle
        return cycle, is_subtour

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if mtz_model:
                return
            print(f"\nCurrent Best Solution ({model.cbGet(GRB.Callback.MIPSOL_OBJ)}):")
            x_sol = model.cbGetSolution(model._x)

            # for key, val in x_sol.items():
            #     if val > 1e-8:
            #         print(*key, val)

            iteration_seen_tours = set()

            selected = gp.tuplelist(a for a in arcs if x_sol[a] > 0.5)
            # print(selected)
            tour_configs = [i for i, _ in selected]
            tour, is_subtour = subtour(selected, tour_configs)

            print(f"Arc numbers: {len(selected)}")

            if len(tour) < len(selected):
                while tour:
                    # add subtour elimination constr. for every pair of cities in subtour
                    if tuple(sorted(tour)) not in iteration_seen_tours:
                        model.cbLazy(
                            gp.quicksum(
                                model._x[i, j] + model._x[j, i]
                                for i, j in combinations(tour, 2)
                                if (i, j) in arcs
                            )
                            <= len(tour) - 1
                        )

                        model.cbLazy(
                            gp.quicksum(
                                model._x[i, j] + model._x[j, i]
                                for i, j in combinations(tour, 2)
                                if (i, j) in arcs
                            )
                            <= len(tour)
                        )

                    # print(sum(x_sol[ci, cj] for ci, cj in combinations(tour, 2)))
                    # print(gp.quicksum(model._x[ci, cj] for ci, cj in combinations(tour+[str(k)], 2))
                    #              <= len(tour))

                    if tuple(sorted(tour)) in model._seen_tours:
                        print("***************** ERROR")
                        print(tour, " seen before")

                    print(f"subtour: {tour}")
                    # print(f"tour_configs : {tour_configs}")
                    # print(f"tour : {tour}")
                    # print(list(combinations(tour, 2)))

                    iteration_seen_tours.add(tuple(sorted(tour)))
                    # print()
                    tour_configs = [i for i, _ in selected]

                    tour_configs = [c for c in tour_configs if c not in tour]
                    tour, is_subtour = subtour(selected, tour_configs)
            else:
                print(f"tour: {tour}")

            model._seen_tours = model._seen_tours.union(iteration_seen_tours)

        elif where == GRB.Callback.MIPNODE:
            return
            # MIP node callback
            print("**** New node ****")
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for key, val in model.cbGetNodeRel(model._x).items():
                    if val > 1e-8:
                        print(*key, val)
                print()

    m._x = x

    # Model Parameters
    m.Params.lazyConstraints = 1

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
    m.optimize(mycallback)

    print(m.Status)
    if m.Status == GRB.INFEASIBLE or m.Status == GRB.CUTOFF:
        return initial_solution

    print(f"Objective: {m.objVal:.2f}")
    tours = []
    selected = gp.tuplelist(a for a in arcs if x[a].x > 0.5)
    for a in arcs:
        if x[a].x > 0.5:
            print(f"x{a}={x[a].x}")
    #
    # for ci in config_pool:
    #     if u[ci].x > 0.5:
    #         print(f"u{ci}={u[ci].x}")

    return 0
