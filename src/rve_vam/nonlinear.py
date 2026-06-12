from __future__ import annotations


class NonlinearMaterialModel:
    """Placeholder for future plasticity/damage material updates."""

    def update_state(self, strain, state):
        raise NotImplementedError("Nonlinear material updates are reserved for a future version.")


class NonlinearSolver:
    """Placeholder for future incremental nonlinear solves."""

    def solve_increment(self, problem):
        raise NotImplementedError(
            "Nonlinear solver is reserved for future plasticity/damage/interface models."
        )
