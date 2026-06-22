#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors
# Copyright 2021 Dustin Kenefake
#
# Modified from qpsolvers/solvers/gurobi_.py to support basis warm-starting.
# gurobi_solve_problem now accepts initvals=(VBasis, CBasis) and returns
# (Solution, bases) instead of Solution.

"""Solver interface for `Gurobi <https://www.gurobi.com/>`__."""

import time
from typing import Optional, Union

import gurobipy
import numpy as np
import scipy.sparse as spa
from gurobipy import GRB

from qpsolvers.problem import Problem
from qpsolvers.solution import Solution


GUROBI_STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


def gurobi_status_name(status_code):
    return GUROBI_STATUS_NAMES.get(status_code, f"UNKNOWN({status_code})")


def qp_constraint_violations(problem, x=None):
    """Return max equality and inequality violations for a candidate x (or zeros)."""
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if x is None:
        x = np.zeros(P.shape[0])
    eq_viol = 0.0
    if A is not None and b is not None:
        eq_viol = float(np.max(np.abs(A @ x - b)))
    ineq_viol = 0.0
    if G is not None and h is not None:
        ineq_viol = float(np.max(G @ x - h))
    return eq_viol, ineq_viol


def diagnose_qp_failure(problem, initvals=None, verbose=True):
    """
    Re-solve a failed QP in-process and return Gurobi status plus IIS info if infeasible.
    """
    model = gurobipy.Model()
    if not verbose:
        model.setParam(GRB.Param.OutputFlag, 0)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = spa.eye(num_vars)
    x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

    if G is not None:
        model.addMConstr(G, x, GRB.LESS_EQUAL, h)
    if A is not None:
        model.addMConstr(A, x, GRB.EQUAL, b)
    if lb is not None:
        model.addMConstr(identity, x, GRB.GREATER_EQUAL, lb)
    if ub is not None:
        model.addMConstr(identity, x, GRB.LESS_EQUAL, ub)

    model.setObjective(0.5 * (x @ P @ x) + q @ x, sense=GRB.MINIMIZE)
    model.setParam("Method", 1)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 3)
    model.update()
    if initvals is not None:
        try:
            model.setAttr("VBasis", model.getVars(), initvals[0])
            model.setAttr("CBasis", model.getConstrs(), initvals[1])
        except gurobipy.GurobiError:
            pass

    model.optimize()
    status = model.status
    report = {
        "status_code": status,
        "status_name": gurobi_status_name(status),
        "num_vars": num_vars,
        "num_eq": 0 if A is None else A.shape[0],
        "num_ineq": 0 if G is None else G.shape[0],
        "iis_constraints": [],
    }

    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        try:
            model.computeIIS()
            for constr in model.getConstrs():
                if constr.IISConstr:
                    report["iis_constraints"].append(constr.ConstrName or f"row_{constr.index}")
        except gurobipy.GurobiError as exc:
            report["iis_error"] = str(exc)

    return report


def gurobi_solve_problem(
    problem: Problem,
    initvals=None,
    verbose: bool = False,
    **kwargs,
):
    """Solve a quadratic program using Gurobi with optional basis warm-start.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start basis as (VBasis, CBasis) lists returned from a previous
        solve, or None to start from scratch.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Tuple of (Solution, (VBasis, CBasis)).
    """
    model = gurobipy.Model()
    if not verbose:
        model.setParam(GRB.Param.OutputFlag, 0)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = spa.eye(num_vars)
    x = model.addMVar(
        num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS
    )

    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None:
        ineq_constr = model.addMConstr(G, x, GRB.LESS_EQUAL, h)
    if A is not None:
        eq_constr = model.addMConstr(A, x, GRB.EQUAL, b)
    if lb is not None:
        lb_constr = model.addMConstr(identity, x, GRB.GREATER_EQUAL, lb)
    if ub is not None:
        ub_constr = model.addMConstr(identity, x, GRB.LESS_EQUAL, ub)
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, sense=GRB.MINIMIZE)
    model.setParam("Method", 1)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 3)
    model.update()
    if initvals is not None:
        model.setAttr("VBasis", model.getVars(), initvals[0])
        model.setAttr("CBasis", model.getConstrs(), initvals[1])
    model.optimize()
    bases = model.getAttr(GRB.Attr.VBasis), model.getAttr(GRB.Attr.CBasis)

    solution = Solution(problem)
    solution.extras["status"] = model.status
    solution.extras["status_name"] = gurobi_status_name(model.status)
    solution.found = model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    if solution.found:
        solution.x = x.X
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)
    else:
        model.update()
    return solution, bases


def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[gurobipy.MConstr],
    eq_constr: Optional[gurobipy.MConstr],
    lb_constr: Optional[gurobipy.MConstr],
    ub_constr: Optional[gurobipy.MConstr],
) -> None:
    solution.z = -ineq_constr.Pi if ineq_constr is not None else np.empty((0,))
    solution.y = -eq_constr.Pi if eq_constr is not None else np.empty((0,))
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = -ub_constr.Pi - lb_constr.Pi
    elif ub_constr is not None:
        solution.z_box = -ub_constr.Pi
    elif lb_constr is not None:
        solution.z_box = -lb_constr.Pi
    else:
        solution.z_box = np.empty((0,))
