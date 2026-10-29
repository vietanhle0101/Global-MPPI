from typing import Any, Literal, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from scipy.stats import qmc
from global_mppi.alg_base import SamplingBasedController, SamplingParams, Trajectory
from global_mppi.risk import RiskStrategy
from global_mppi.task_base import Task
from mujoco import mjx
from ksos_tools.solvers import ksos
from jax import lax

KSOS_SAMPLING_MODES = ("uniform", "gaussian", "linspace", "sobol")

@dataclass
class GlobalMPPIParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """
    iter: jax.Array
    lse_cost: jax.Array
    cost: jax.Array
    knots: jax.Array
    ksos_mean: jax.Array
    ksos_result_cost: jax.Array
    ksos_trace: jax.Array

class GlobalMPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        ctrl_name : str = "globalmppi",
        ksos_solver: str = "newton-rs",
        ksos_sampling: str = "uniform",
        ksos_num_samples: int = 256,
        elite_num: int = 80,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
            ctrl_name=ctrl_name,
        )
        # mppi parameters
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        
        # lse parameters
        self.lse_num_samples = 100
        self.lse_smoothing_sigma = jnp.array([0.3, 0.2, 0.1, 0.05, 0.001])
        self.lse_lambda = 0.1

        # ksos parameters
        self.ksos_num_samples = ksos_num_samples
        self.elite_num = elite_num
        self.ksos_lambda = 1e-5
        self.ksos_sigma = 0.01
        self.ksos_epsilon = 1e-2
        self.ksos_kernel = "Laplace"
        self.ksos_iterations = 100
        self.ksos_linesearch = True
        self.ksos_num_restart = 5
        self.ksos_solver = ksos_solver
        if ksos_sampling == "flow":
            raise NotImplementedError(
                "ksos_sampling='flow' is not implemented yet."
            )
        if ksos_sampling not in KSOS_SAMPLING_MODES:
            raise ValueError(
                f"Unknown ksos_sampling mode: {ksos_sampling!r} "
                f"(expected one of {KSOS_SAMPLING_MODES + ('flow',)})"
            )
        self.ksos_sampling = ksos_sampling
        if self.ksos_sampling == "sobol":
            # Precomputed once (outside jit) since scipy's QMC sampler isn't
            # traceable; `sample_ksos_knots` only does elementwise jax ops on
            # this fixed unit-cube point set.
            dim = num_knots * task.model.nu
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            self.sobol_unit = jnp.asarray(sampler.random(self.ksos_num_samples))
        self.decay_rate = 0.80
        self.is_lse_smoothing = True

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> GlobalMPPIParams:
        """Initialize the policy parameters, extending the base params with the
        extra fields GlobalMPPI needs to track the KSOS/LSE optimization state
        (iteration counter, LSE-smoothed costs, KSOS mean and trace)."""
        _params = super().init_params(initial_knots, seed)
        iter = 0
        lse_cost = jnp.array(0.0)
        cost = jnp.array(0.0)
        ksos_mean = jnp.zeros_like(_params.mean)
        ksos_result_cost = jnp.array(0.0)
        return GlobalMPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, iter=iter, lse_cost=lse_cost, cost = cost, ksos_mean=ksos_mean, ksos_result_cost = ksos_result_cost, knots=None, best_cost= _params.best_cost, ksos_trace=None,best_trace=_params.best_trace)

    def sample_knots(self, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """Sample MPPI candidate knots as Gaussian noise around the current
        mean, with the noise scale taken from `lse_smoothing_sigma` at the
        current iteration (used by the base class's `optimize`/MPPI update,
        as opposed to the KSOS candidate sampling in `sample_ksos_knots`)."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        sigma = self.lse_smoothing_sigma[params.iter]
        controls = params.mean + sigma * noise
        jax.debug.print("sigma: {}", sigma)

        return controls, params.replace(rng=rng)

    def sample_ksos_knots(self, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """Sample `ksos_num_samples` candidate knot sequences for the KSOS
        solver. Candidates are drawn from a trust region centered at the
        previous KSOS mean (or zero on the first iteration) whose radius
        shrinks geometrically with `decay_rate` each iteration. The sampling
        distribution is controlled by `self.ksos_sampling`
        ("linspace", "uniform", "gaussian", or "sobol")."""
        rng, sample_rng = jax.random.split(params.rng)

        center = lax.cond(
            jnp.equal(params.iter, 0),
            lambda _: jnp.zeros_like(params.ksos_mean),
            lambda _: params.ksos_mean,
            operand=None,
        )
        radius = (
            jnp.power(self.decay_rate, params.iter).astype(params.mean.dtype)
            * self.task.u_max
        )
        if self.ksos_sampling == "linspace":
            ksos_knots_old = jnp.linspace(
                center - radius,
                center + radius,
                self.ksos_num_samples,
            )
            # Shape: [ksos_num_samples, num_knots, nu]
            ksos_knots_old = ksos_knots_old.reshape(-1, 2)     # shape (24, 2)
            np.random.shuffle(ksos_knots_old)
            ksos_knots_old = ksos_knots_old.reshape(30, 6, 2)
        elif self.ksos_sampling == "uniform":
            ksos_knots = jax.random.uniform(
                sample_rng,
                (
                    self.ksos_num_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),                          
                minval=center - radius,
                maxval=center + radius,
            )
        elif self.ksos_sampling == "gaussian":
            noise = jax.random.normal(
                sample_rng,
                (
                    self.ksos_num_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),
            )
            ksos_knots = center + radius * noise
        elif self.ksos_sampling == "sobol":
            dim = int(center.size)
            u = self.sobol_unit.astype(params.mean.dtype)  # (N, dim) in [0,1)
            shift = jax.random.uniform(sample_rng, (dim,), dtype=u.dtype)
            u = jnp.mod(u + shift, 1.0)

            lower = (center - radius).reshape(-1)
            upper = (center + radius).reshape(-1)

            samples = lower + (upper - lower) * u   # (N, dim)

            ksos_knots = samples.astype(params.mean.dtype).reshape(
                self.ksos_num_samples, self.num_knots, self.task.model.nu
            )
        else:
            raise ValueError(f"Unknown ksos_sampling mode: {self.ksos_sampling}")
        
        return ksos_knots, params.replace(rng=rng)
    
    def sample_lse_knots(self, ksos_knots: jax.Array, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """For each KSOS candidate knot sequence, sample `lse_num_samples`
        nearby perturbations (Gaussian noise scaled by the current
        `lse_smoothing_sigma`) used to estimate a smoothed cost for that
        candidate via log-sum-exp in `lse_smoothing_rollout`."""
        rng, sample_rng = jax.random.split(params.rng)
        num_ksos = ksos_knots.shape[0]
        noise = jax.random.normal(
            sample_rng,
            (
                num_ksos,
                self.lse_num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
            dtype=params.mean.dtype,
        )

        sigma = self.lse_smoothing_sigma[params.iter]
        lse_knots = ksos_knots[:, None, :, :] + sigma * noise       #Shape: [ksos_num_samples, lse_num_samples, num_knots, nu] same as [batch_size, num_samples, N]
        
        return lse_knots, params.replace(rng=rng)

    def general_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """Roll out each KSOS candidate knot sequence once (no domain
        randomization, no LSE smoothing) and record its total cost. This is
        the cheaper alternative to `lse_smoothing_rollout`, used when
        `self.is_lse_smoothing` is False."""

        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)
        _, rollouts = self.eval_rollouts(self.model, state, controls, knots)
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        return params.replace(knots= knots, lse_cost=costs, cost = rollouts.costs, ksos_trace=rollouts.trace_sites)
    
    def lse_smoothing_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """Estimate a smoothed cost for each KSOS candidate knot sequence by
        rolling out `lse_num_samples` nearby perturbations (from
        `sample_lse_knots`) and combining their costs with a log-sum-exp soft
        minimum, tempered by `lse_lambda`. This trades extra rollouts for a
        cost estimate that is less sensitive to a single unlucky sample than
        `general_rollout`."""

        # generate lse samples
        lse_knots, params = self.sample_lse_knots(knots, params)
        
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        num_rollouts, num_lse, num_knots, nu = lse_knots.shape

        flat_knots = lse_knots.reshape(num_rollouts * num_lse, num_knots, nu)
        flat_controls = self.interp_func(tq, tk, flat_knots)  # (num_rollouts, H, nu)
        _, flat_rollouts = self.eval_rollouts(self.model, state, flat_controls, flat_knots)
        total_costs = flat_rollouts.costs.reshape(num_rollouts, num_lse, flat_rollouts.costs.shape[1])
        
        # sum cost over time steps
        sum_costs = jnp.sum(total_costs, axis=-1)  # sum over time steps
        min_costs = jnp.min(sum_costs, axis=1, keepdims=True)
        samples_exp = jnp.exp( - (sum_costs - min_costs) / self.lse_lambda )   # Shape: [num_rollouts, num_lse]
        sum = jnp.mean(samples_exp, axis=1, keepdims=True)
        lse_cost = - jnp.log(sum) * self.lse_lambda + min_costs

        return params.replace(knots= knots, cost = total_costs, lse_cost=lse_cost.squeeze())
    
    def ksos_rollout(self, state: mjx.Data, params: Any) -> Any:
        """Warm-start the spline knot times to the current sim time, sample a
        batch of KSOS candidate knots, and roll them all out to get a cost
        for each candidate (via `lse_smoothing_rollout` or `general_rollout`,
        depending on `self.is_lse_smoothing`). The resulting per-candidate
        costs are consumed by `solve_ksos` to fit an updated mean.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters, with `knots`/`lse_cost`/`cost` set to
            the sampled candidates and their evaluated costs.
        """
        # update tk with sim time
        tk = params.tk
        new_tk = (jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time)
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        # sample knots
        knots, params = self.sample_ksos_knots(params)
        knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

        rng, dr_rng = jax.random.split(params.rng)
        
        # compute lse smoothed costs
        if self.is_lse_smoothing:
            params = self.lse_smoothing_rollout(state, tk, knots, params)
        else:
            params = self.general_rollout(state, tk, knots, params)

        params = params.replace(rng=rng)

        return params
    
    def solve_ksos(
        self,
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """Fit an updated mean from the KSOS candidates rolled out in
        `ksos_rollout`. Takes the `self.elite_num` lowest-cost candidates,
        drops any whose per-step cost trace never decreases (a heuristic filter for rollouts
        that never make progress), and hands the survivors to the KernelSOS
        solver (`ksos.solve`, using `self.ksos_solver`) to produce a new
        `ksos_mean`. Falls back to returning `params` unchanged if every
        elite is filtered out or the solver fails."""

        indices = jnp.argsort(params.lse_cost)
        elites = indices[: self.elite_num]

        if self.is_lse_smoothing == False:
            costs = params.cost[elites]
        else:
            costs = params.cost.mean(axis=1)[elites]
        lse_costs = params.lse_cost[elites]
        knots_elites = params.knots[elites]

        # filter out bad rollouts
        first_20 = costs[:, :20]
        diff = first_20[:, 1:] - first_20[:, :-1]  
        rollout_index = (diff < 0).any(axis=1).astype(int)   
        idx_ones = jnp.where(rollout_index == 1)[0]

        if idx_ones.shape[0] == 0:
            print("All elites are bad rollouts !!!")
            return params
        
        costs = costs[idx_ones]
        lse_costs = lse_costs[idx_ones]
        knots_elites = knots_elites[idx_ones]

        n_samples = lse_costs.shape[0]
        samples = np.asarray(
            knots_elites.reshape(n_samples, -1),
            dtype=np.float64,
        )
        sample_costs = np.asarray(lse_costs, dtype=np.float64).reshape(-1)
        try:
            updated_knots, info = ksos.solve(
                f=None,
                samples=samples,
                f_samples=sample_costs,
                lambd=self.ksos_lambda,
                sigma=float(self.ksos_sigma),
                epsilon=self.ksos_epsilon,
                kernel=self.ksos_kernel,
                solver=self.ksos_solver,
            )
            if updated_knots is None:
                raise RuntimeError(
                    info.get("status", "KSOS solver returned no solution")
                )
            updated_knots = np.asarray(
                updated_knots,
                dtype=np.float32,
            ).reshape(
                self.num_knots,
                self.task.model.nu,
            )
        except Exception as error:
            print(f"KSOS solver failed: {error}")
            return params

        return params.replace(
            ksos_mean=updated_knots,
            ksos_result_cost=info["cost"],
            lse_cost=lse_costs,
        )

    def update_params(
        self, params: GlobalMPPIParams, rollouts: Trajectory
    ) -> GlobalMPPIParams:
        """MPPI update: set the mean to a softmax-weighted average of the
        rollout knots, favoring lower-cost samples (temperature-scaled by
        `self.temperature`). This is the local-search step the base class's
        `optimize` runs after `solve_ksos` seeds the mean with a KSOS
        solution."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mppi_mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        
        return params.replace(mean=mppi_mean)
