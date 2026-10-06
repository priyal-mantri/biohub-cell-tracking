"""
Open-Source MILP Solver Wrapper using SciPy (HiGHS MILP Solver).
"""

from typing import Tuple, Optional
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

from src.optimization.base import SolverResult, OptimizationConfig
from src.optimization.constraints import ConstraintMatrix


class MILPSolverWrapper:
    """
    Transparent solver wrapper for binary mixed-integer linear programming (MILP).

    Employs the open-source HiGHS solver integrated natively in scipy.optimize.milp.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def solve(
        self,
        c_min: np.ndarray,  # (E,) minimization cost vector (-w_e)
        constraints: ConstraintMatrix
    ) -> Tuple[np.ndarray, SolverResult]:
        """
        Solves binary MILP optimization problem:
            Minimize   c_min^T x
            Subject to lb <= A x <= ub
                       x_e in {0, 1}

        :param c_min: Cost vector shape (E,)
        :param constraints: ConstraintMatrix object
        :return: Tuple of (binary solution vector x_e shape (E,) float64, SolverResult)
        """
        E = len(c_min)
        if E == 0:
            result = SolverResult(
                status="optimal",
                success=True,
                objective_value=0.0,
                solve_time_sec=0.0,
                num_variables=0,
                num_constraints=0,
                selected_edge_count=0,
                termination_reason="empty_graph"
            )
            return np.empty(0, dtype=np.float64), result

        # Binary integrality constraints (1 = integer/binary)
        integrality = np.ones(E, dtype=np.int32)
        # Variable bounds [0, 1]
        bounds = Bounds(np.zeros(E, dtype=np.float64), np.ones(E, dtype=np.float64))

        if constraints is not None and constraints.matrix.shape[0] > 0:
            scipy_constraints = constraints.to_scipy_constraint()
        else:
            scipy_constraints = None
        options = {"time_limit": max(1.0, float(self.config.timeout_sec))}

        start_time = time.time()
        res = milp(
            c=c_min,
            integrality=integrality,
            bounds=bounds,
            constraints=scipy_constraints,
            options=options
        )
        solve_time = float(time.time() - start_time)

        # Interpret status codes
        status_map = {
            0: "optimal",
            1: "iteration_limit",
            2: "time_limit",
            3: "infeasible",
            4: "unbounded",
        }
        status_str = status_map.get(res.status, f"unknown_code_{res.status}")
        term_reason = res.message if hasattr(res, "message") else status_str

        x_sol = np.zeros(E, dtype=np.float64)
        if res.x is not None:
            # Round binary variables cleanly
            x_sol = np.round(np.clip(res.x, 0.0, 1.0)).astype(np.float64)

        selected_count = int(np.sum(x_sol == 1.0))
        # Maximization objective value is -c_min^T x = -res.fun
        max_obj_val = float(-res.fun) if res.fun is not None else 0.0

        solver_result = SolverResult(
            status=status_str,
            success=bool(res.success or res.status == 0),
            objective_value=max_obj_val,
            solve_time_sec=solve_time,
            num_variables=E,
            num_constraints=constraints.matrix.shape[0] if (constraints is not None and constraints.matrix is not None) else 0,
            selected_edge_count=selected_count,
            termination_reason=term_reason
        )

        return x_sol, solver_result
