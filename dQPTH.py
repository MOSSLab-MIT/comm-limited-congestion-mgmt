import numpy as np
import torch.multiprocessing as mp

import torch
from torch import nn

import qpsolvers
from scipy.sparse import csc_matrix, bmat
import torch
import scipy.sparse as sp
import qdldl

import sys
import os
import warnings

src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(src_dir)
sys.path.append(parent_dir)

from dQPTH_helpers.set_solver_tolerance import set_solver_tolerance
from dQPTH_helpers.gurobi_ws import (
    gurobi_solve_problem as gurobi_ws_solve_problem,
    diagnose_qp_failure,
    gurobi_status_name,
    qp_constraint_violations,
)

import dQPTH_helpers.sparse_helper as sparse_helper
import dQPTH_helpers.lin_solvers as lin_solvers
import time


def _schur_d_vals(slacks, lams, eps=1e-8):
    """Diagonal D_s entries (s / lambda) for the Schur complement top-left block."""
    s_safe = torch.clamp(slacks.flatten(), min=eps)
    lam_safe = torch.clamp(lams.flatten(), min=eps)
    return (s_safe / lam_safe).detach().numpy()


def _schur_blocks_from_diagonal_q(scipy_G, scipy_A, Q_inv_diag):
    """
    Build fixed Schur blocks when Q is diagonal.

    G Q^{-1} G^T is G @ diag(Q_inv) @ G^T; avoid materializing diag(Q_inv)
    by column-scaling G (and A) once.
    """
    G_scaled = scipy_G.multiply(Q_inv_diag)
    A_scaled = scipy_A.multiply(Q_inv_diag)
    S_GG = scipy_G @ G_scaled.T
    S_GA = scipy_G @ A_scaled.T
    S_AA = scipy_A @ A_scaled.T
    return S_GG, S_GA, S_AA


def _assemble_schur_csc(S_GG, S_GA, S_AA, d_vals):
    """Assemble the full Schur matrix S with D_s on the top-left diagonal."""
    n_ineq = S_GG.shape[0]
    S = sp.bmat(
        [
            [S_GG + sp.diags(d_vals), S_GA],
            [S_GA.T, S_AA],
        ],
        format="csc",
    )
    return S, _top_left_diag_data_indices(S, n_ineq)


def _top_left_diag_data_indices(S_csc, n_ineq):
    """Indices into S_csc.data for the top-left block diagonal (one per row)."""
    diag_data_idx = np.empty(n_ineq, dtype=np.int64)
    for row in range(n_ineq):
        start, end = S_csc.indptr[row], S_csc.indptr[row + 1]
        cols = S_csc.indices[start:end]
        pos = np.flatnonzero(cols == row)
        if pos.size != 1:
            raise ValueError(
                f"Expected one diagonal entry in Schur row {row}, found {pos.size}"
            )
        diag_data_idx[row] = start + pos[0]
    return diag_data_idx


class dQPTH_layer(nn.Module):
    '''solves and differentiates
    x^* = argmin_x 1/2 x^T Q x + q^T x
             s.t.  G x <= h
                   A x  = b
    (including dual variables mu^*,nu^*)

    Written by James Chen, based on dQP and qpth code, hence our name dQPTH

    We use much of the code from dQP, but adapt the backward pass from qpth, as we find that the Schur complement trick
    allows for better stability. (We do not claim this to be a limitation of dQP, but rather a limitation of the author's
    ability to implement and tune numerical algorithms.) For the forward pass we default to using Gurobi.

    dQP and qpth allow for batched QPs with the same size. We instead allow for multiprocessing of QPs of different
    sizes, and assign each QP to a different worker.
    '''
    def __init__(self,settings=None,workers=3,pool=None):
        super().__init__()

        if settings is None:
            settings = build_settings() # call with defaults

        for k, v in settings.items():
            setattr(self, k, v)

        self.dim = None
        self.nIneq = None
        self.nEq = None
        self.nBatch = None

        self.differentiate_QP = differentiate_QP.apply # set-up differentiation through active constraints

        # initialize numpy variables which are carried implicitly ; torch version is not stored
        self.x_star_np = None
        self.mu_star_np = None
        self.nu_star_np = None
        self.x_star_np_cache = dict()

        self.Q_np = None
        self.q_np = None
        self.G_np = None
        self.h_np = None
        self.A_np = None
        self.b_np = None

        self.backward_cache = dict()

        if pool == None:
            self.pool = mp.Pool(processes=workers)
            print(f"Persistent multiprocessing pool initialized with {workers} workers.")
        else:
            self.pool = pool

    def close(self):
        """Method to close the pool gracefully when done with the object."""
        print("Closing persistent multiprocessing pool.")
        self.pool.close()
        self.pool.join() # Wait for workers to finish current tasks and exit

    def forward(self, Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t):
        # check shapes, extract dim,nIneq,nEq,nBatch
        Q,q,G,h,A,b = self.get_shapes(Q, q, G, h, A, b)
        self.data_to_np(Q, q, G, h, A, b)
        self.solve(t)
        self.nu_star_with_inactive_np = self.nu_star_np.copy()
        x_star = self.setup_diff(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)

        return x_star


    def clear_cache(self):
        """
        Clears cache of warm-start points.
        """
        self.x_star_np_cache = dict()
        self.backward_cache = dict()


    def get_shapes(self,Q,q,G,h,A,b):
        '''
        extract dim,nIneq,nEq,nBatch and standardize shapes
        note, these checks will certainly overlap with checks in qpsolvers and their calls
        can find ways to improve the last dimension = 1 for vectors condition ; may or may not need to rewrite other code
        another option is just to let other parts fail and just extract, without checking
        '''
        # assume solve_type is sparse and that Q, G are lists
        self.nBatch = len(Q)
        self.nIneq = list()
        self.nEq = list()
        self.dim = list()
        for i in range(self.nBatch):
            nIneq, dim = G[i].size()
            self.nIneq.append(nIneq)
            self.dim.append(dim)

            if A[i] is not None and b[i] is not None:
                nEq, _ = A[i].size()
                self.nEq.append(nEq)
            else:
                self.nEq.append(0)

        return Q,q,G,h,A,b


    def data_to_np(self,Q,q,G,h,A,b):
        '''
        Convert torch QP parameters to numpy and scipy variables, accounting for sparsity
        Stores variables in the class and optionally saves them
        '''
        self.Q_np = []
        self.G_np = []
        if A is not None:
            self.A_np = []
            for i in range(self.nBatch):
                self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]
                self.A_np += [sparse_helper.csc_torch_to_scipy(A[i])]
        else:
            self.A_np = None
            for i in range(self.nBatch):
                self.Q_np += [sparse_helper.csc_torch_to_scipy(Q[i])]
                self.G_np += [sparse_helper.csc_torch_to_scipy(G[i])]

        self.q_np = []
        self.h_np = []
        for i in range(self.nBatch):
            self.q_np += [q[i].detach().numpy()]
            self.h_np += [h[i].detach().numpy()]

        self.b_np = []
        for i in range(self.nBatch):
            if b[i] is not None:
                self.b_np += [b[i].detach().numpy()]
            else:
                self.b_np += [None]

        return


    def reset_cache(self):
        self.x_star_np_cache = dict()


    def solve(self, t):
        '''
        Solve the QP using qpsolvers
        data_to_np must be called before
        '''

        # initialize
        self.x_star_np = [None for _ in range(self.nBatch)]
        if self.nEq[0] > 0:
            self.mu_star_np = [None for _ in range(self.nBatch)]
        self.nu_star_np = [None for _ in range(self.nBatch)]

        mp.set_start_method('spawn', force=True)

        # start_time = time.time()
        qp_solve_args = []
        for i in range(self.nBatch):
            if self.warm_start_from_previous and (i,0) in self.x_star_np_cache.keys(): # false if nBatch > 1
                if (i,t) in self.x_star_np_cache.keys():
                    initvals = self.x_star_np_cache[(i,t)]
                else:
                    initvals = self.x_star_np_cache[(i,t-1)]
            else:
                initvals = None
            if self.nEq[i] > 0:
                kwargs_main = {
                    "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                 h=self.h_np[i], A=self.A_np[i], b=self.b_np[i]),
                    "solver": self.qp_solver,
                    "verbose": self.verbose,
                    "initvals": initvals
                }
            else:
                kwargs_main = {
                    "problem": qpsolvers.Problem(P=self.Q_np[i], q=self.q_np[i], G=self.G_np[i],
                                                    h=self.h_np[i], A=None, b=None),
                    "solver": self.qp_solver,
                    "verbose": self.verbose,
                    "initvals": initvals
                }
            qp_solve_args.append((self.qp_solver_keywords, kwargs_main))
        results = self.pool.starmap(call_qp_solver_pool_target, qp_solve_args)
        for i in range(self.nBatch):
            x, y, z, V_basis, C_basis, found, extras = results[i]

            if x is None or not found:
                self._report_solver_failure(i, t, qp_solve_args[i][1], extras)
                raise RuntimeError(
                    f"QP solver failed for agent {i} at rollout step t={t}: "
                    f"{extras.get('status_name', 'unknown')}"
                )

            self.x_star_np[i] = x
            # implicitly assumes if nBatch > 1 then duals are available:
            self.mu_star_np[i] = y
            self.nu_star_np[i] = z
            self.x_star_np_cache[(i,t)] = (V_basis, C_basis)
        return None


    def _report_solver_failure(self, agent_idx, t, kwargs_main, extras):
        problem = kwargs_main["problem"]
        initvals = kwargs_main.get("initvals")
        status_name = extras.get("status_name", gurobi_status_name(extras.get("status", -1)))
        eq_viol, ineq_viol = qp_constraint_violations(problem)

        print("\n" + "=" * 72)
        print(f"Gurobi/QP solve failed")
        print(f"  agent index : {agent_idx}")
        print(f"  rollout step: t={t}")
        print(f"  status      : {status_name} (code {extras.get('status', 'n/a')})")
        print(f"  problem size: nVar={problem.P.shape[0]}, nEq={0 if problem.A is None else problem.A.shape[0]}, "
              f"nIneq={0 if problem.G is None else problem.G.shape[0]}")
        print(f"  violations at x=0: max|Ax-b|={eq_viol:.3e}, max(Gx-h)={ineq_viol:.3e}")
        if initvals is not None:
            print("  warm start  : basis from previous step was supplied")
        else:
            print("  warm start  : none")

        diag = diagnose_qp_failure(problem, initvals=initvals, verbose=True)
        print(f"  diagnostic re-solve status: {diag['status_name']} (code {diag['status_code']})")
        if diag["iis_constraints"]:
            print(f"  IIS constraints ({len(diag['iis_constraints'])} in conflict):")
            for name in diag["iis_constraints"][:20]:
                print(f"    - {name}")
            if len(diag["iis_constraints"]) > 20:
                print(f"    ... and {len(diag['iis_constraints']) - 20} more")
        if "iis_error" in diag:
            print(f"  IIS error: {diag['iis_error']}")
        print("=" * 72 + "\n")


    def setup_diff(self, csc_Q, q, csc_G, G, h, csc_A, b, scipy_Q, scipy_G, scipy_A, t):
        '''
        Form the reduced KKT and set-up derivatives through x_star
        Sets non_differentiable check
        '''
        x_star_list = []
        for i in range(self.nBatch):
            zhats = torch.tensor(self.x_star_np[i])
            slacks = h[i] - G[i] @ zhats
            lams = torch.tensor(self.nu_star_np[i]).unsqueeze(0)
            d_vals = _schur_d_vals(slacks, lams)

            if (i, "S_GG") not in self.backward_cache:
                eps = 1e-7
                Q_inv_diag = 1.0 / (scipy_Q[i].diagonal() + eps)
                S_GG, S_GA, S_AA = _schur_blocks_from_diagonal_q(
                    scipy_G[i], scipy_A[i], Q_inv_diag
                )

                self.backward_cache[(i, "S_GG")] = S_GG
                self.backward_cache[(i, "S_GA")] = S_GA
                self.backward_cache[(i, "S_AA")] = S_AA
                self.backward_cache[(i, "S_GG_diag")] = np.asarray(S_GG.diagonal()).ravel()

                S, diag_data_idx = _assemble_schur_csc(S_GG, S_GA, S_AA, d_vals)
                factor = qdldl.Solver(S)
                self.backward_cache[(i, "S_csc")] = S
                self.backward_cache[(i, "diag_data_idx")] = diag_data_idx
                self.backward_cache[(i, "factor")] = factor
                self.backward_cache[(i, "indices")] = S.indices.copy()
                self.backward_cache[(i, "indptr")] = S.indptr.copy()
            else:
                S_GG_diag = self.backward_cache[(i, "S_GG_diag")]
                S = self.backward_cache[(i, "S_csc")]
                diag_data_idx = self.backward_cache[(i, "diag_data_idx")]
                S.data[diag_data_idx] = S_GG_diag + d_vals

                factor = self.backward_cache[(i, "factor")]
                factor.update(S)
                cached_indices = self.backward_cache[(i, "indices")]
                cached_indptr = self.backward_cache[(i, "indptr")]

                if not (np.array_equal(S.indices, cached_indices) and
                        np.array_equal(S.indptr, cached_indptr)):
                    print(f"ERROR: Sparsity pattern changed at batch index {i}!")
                    print(f"Old nnz: {len(cached_indices)}, New nnz: {len(S.indices)}")

            diff_params = {
                "x_star": torch.tensor(self.x_star_np[i]),
                "eq_dual_star": torch.tensor(self.mu_star_np[i]),
                "ineq_dual_star": torch.tensor(self.nu_star_np[i]),
                "scipy_Q": scipy_Q[i],
                "scipy_G": scipy_G[i],
                "scipy_A": scipy_A[i],
                "nonsparse_G": G[i],
                "dim": self.dim[i],
                "nEq": self.nEq[i],
                "nIneq": self.nIneq[i],
                "agent": i,
                "t": t,
                "warm_start_from_previous": self.warm_start_from_previous,
                "other_factor": self.backward_cache[(i, "factor")]
            }

            # torch.index_select and .values() only work with COO sparse matrices
            csc_Q_i = csc_Q[i].to_sparse_coo().coalesce().values()
            q_i = q[i]
            csc_G_i = csc_G[i].to_sparse_coo().coalesce().values()

            if self.nEq[i] > 0:
                csc_A_i = csc_A[i].to_sparse_coo().coalesce().values()
                b_i = b[i]
            else:
                csc_A_i = None
                b_i = None

            h_i = h[i]
            x_star = self.differentiate_QP(csc_Q_i, q_i, csc_G_i, h_i, csc_A_i, b_i, diff_params)

            x_star_list += [x_star]

        return x_star_list


def call_qp_solver_pool_target(qp_solver_keywords, kwargs_main):
    """
    Worker function for the multiprocessing pool.
    Returns results (numpy arrays/lists) which the pool gathers.
    """
    if kwargs_main.get("solver") == "gurobi":
        solution, bases = gurobi_ws_solve_problem(
            problem=kwargs_main["problem"],
            initvals=kwargs_main.get("initvals"),
            verbose=kwargs_main.get("verbose", False),
            **qp_solver_keywords,
        )
    else:
        full_kwargs = dict(**kwargs_main, **qp_solver_keywords)
        solution = qpsolvers.solve_problem(**full_kwargs)
        bases = (None, None)

    extras = dict(getattr(solution, "extras", {}) or {})
    if "status" in extras and "status_name" not in extras:
        extras["status_name"] = gurobi_status_name(extras["status"])
    return solution.x, solution.y, solution.z, bases[0], bases[1], solution.found, extras


class differentiate_QP(torch.autograd.Function):
    '''
    Differentiate the QP explicitly using the reduced KKT formed with the active constraints
    '''

    @staticmethod
    def forward(ctx, Q, q, G, h, A, b, params):
        '''
        Just return known solution as differentiable parameter. Store data for backwards.
        '''
        zhats = params["x_star"]
        ctx.zhats = zhats
        ctx.lams = params["ineq_dual_star"]
        ctx.nus = params["eq_dual_star"]
        ctx.scipy_Q = params["scipy_Q"]
        ctx.scipy_G = params["scipy_G"]
        ctx.scipy_A = params["scipy_A"]
        nonsparse_G = params["nonsparse_G"]
        ctx.slacks = h - nonsparse_G @ zhats
        ctx.other_factor = params["other_factor"]
        return zhats

    @staticmethod
    def backward(ctx, dl_dzhat):
        scipy_G = ctx.scipy_G
        scipy_A = ctx.scipy_A
        other_factor = ctx.other_factor

        # Q is diagonal, so 1/Q is a vector and "Q_inv @ v" is an elementwise scale.
        eps = 1e-7
        Q_inv_diag = 1.0 / (ctx.scipy_Q.diagonal() + eps)

        # RHS for the Schur system. Eliminating dx gives
        #   rhs_schur = [-G Q_inv rhs_x ; -A Q_inv rhs_x].
        if torch.is_tensor(dl_dzhat):
            rhs_x_np = dl_dzhat.detach().cpu().numpy()
        else:
            rhs_x_np = np.asarray(dl_dzhat)
        invQ_rhs_x = Q_inv_diag * rhs_x_np
        rhs_schur = np.concatenate([
            -(scipy_G @ invQ_rhs_x),
            -(scipy_A @ invQ_rhs_x),
        ])

        # The Schur factor was already built (or updated) for this timestep
        # in setup_diff and stashed in ctx.other_factor, so reuse it rather
        # than rebuilding S = [[GQ^-1 G^T + D_s, GQ^-1 A^T],
        #                     [AQ^-1 G^T,        AQ^-1 A^T]] and re-factoring.
        d_dual = other_factor.solve(rhs_schur)
        n_ineq = scipy_G.shape[0]
        dhs = torch.tensor(-d_dual[:n_ineq])
        dbs = torch.tensor(-d_dual[n_ineq:])

        # Only dhs and dbs are needed for our use case; the others are skipped.
        return (None, None, None, dhs, None, dbs, None)


def build_settings(check_PSD=False,time=False,solve_type="dense",dual_available=None,normalize_constraints=False,empty_batch=True,warm_start_from_previous=False,omp_parallel=False,n_cpu=None, # general arguments
                   refine_active=False, # active arguments
                   qp_solver=None,verbose=False,qp_solver_keywords=None,eps_abs=1e-6,eps_rel=0, # qp solver arguments ... default to solver preference
                   lin_solver=None): # linear solver arguments

    available_qp_solvers = qpsolvers.available_solvers

    if verbose:
        print("Available QP solvers:\n" + str(qpsolvers.available_solvers))
        print("Available linear solvers:\n" + str(lin_solvers.get_dense_solvers() + lin_solvers.get_sparse_solvers() + qpsolvers.available_solvers))

    if solve_type == "dense":
        if qp_solver is None: # get first available qp solver
            if "cvxopt" in qpsolvers.dense_solvers:
                qp_solver = "cvxopt"
            else:
                qp_solver = qpsolvers.dense_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy LU"
        # assert(qp_solver in qpsolvers.dense_solvers) # qpsolvers labeling of dense/sparse is not a strict classification
        assert(lin_solver in lin_solvers.get_dense_solvers())
    elif solve_type == "sparse":
        if qp_solver is None: # get first available qp solver
            if "gurobi" in qpsolvers.sparse_solvers:
                qp_solver = "gurobi"
            else:
                qp_solver = qpsolvers.sparse_solvers[0]
        if lin_solver is None:
            lin_solver = "scipy SPLU"
        # assert(qp_solver in qpsolvers.sparse_solvers)
        assert(lin_solver in lin_solvers.get_sparse_solvers())

    if dual_available is None:
        if qp_solver in ["clarabel","cvxopt","daqp","ecos","gurobi","highs","hpipm","mosek","osqp","piqp","proxqp","qpalm","qpoases","qpswift","quadprog","scs"]:
            if verbose:
                print("Solver is in base qpsolvers, duals are available.")
            dual_available = True
        else:
            raise("Solver is not in base qpsolvers, user must specify (T/F) if duals are available")


    if omp_parallel and n_cpu is None:
        n_cpu = os.cpu_count()
        print("No CPU count given, using all available: " + str(n_cpu))

    if qp_solver_keywords is None:
        qp_solver_keywords = {}
        qp_solver_keywords = set_solver_tolerance(qp_solver_keywords,qp_solver,eps_abs,eps_rel)

    else:
        print("eps_abs,eps_rel ignored if custom keywords given")

    if verbose:
        print("qp_solver: " + str(qp_solver))
        print("lin_solver: " + str(lin_solver))
        print("qp_solver_keywords: " + str(qp_solver_keywords))

    assert(isinstance(qp_solver_keywords,dict))

    settings = {
        "verbose" : verbose,
        "check_PSD" : check_PSD,
        "time" : time,
        "available_qp_solvers" : available_qp_solvers,
        "solve_type" : solve_type,
        "qp_solver": qp_solver,
        "dual_available": dual_available,
        "normalize_constraints": normalize_constraints,
        "empty_batch": empty_batch,
        "warm_start_from_previous": warm_start_from_previous,
        "qp_solver_keywords" : qp_solver_keywords,
        "lin_solver": lin_solver,
        "refine_active": refine_active,
        "omp_parallel" : omp_parallel,
        "n_cpu" : n_cpu
    }

    return settings
