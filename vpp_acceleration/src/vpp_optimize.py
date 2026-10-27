"""
Search of the section that maximizes the boat speed, with the VPP-targeted sequential co-kriging
of the foil torsor (real runs of 2026-10-22: NPLLT x NeuralFoil / AVL x XFOIL or NPLLT x XFOIL,
MDS VPP, rigid or flexible foil).

The problem (RealFoilProblem). A section = 16 Kulfan weights of a KulfanBoxFamily (no crossing
surfaces, NeuralFoil confident, Ixx constraint), reduced to its k first PCA scores. Inputs of the
surrogate: (z_1..z_k, Cant_Eff, Rake_Eff, Yaw_Eff), no Hvol. Outputs: Fx, Fy, Fz, Mx, My, Mz, Cpmin
(7 co-krigings, rho bounded v2).

The objective. The mean, over the navigation conditions, of the boat speed given by the MDS VPP
whose foil is the mean of the co-kriging (vpp_coupling), plus the deformation correction of the
flexible foil when the VPP has one (vpp_coupling.FSICorrection, third run). The conditions can be
the nodes of an operating profile (src/operating_profile.py), each with its probability w_c: the
mean is then weighted, sum_c w_c V_c, over the conditions solved for every compared section.

The loop (VPPSectionOptimizer), every `update_every` batches:
  1. screening: for n_screen sections and the incumbent, the speed change predicted to FIRST
     ORDER at the attitude of the incumbent's states, condition c:
         dV_c(z) = sum_k (dV/dC_k)_c [C_k(z, att_c) - C_k(z_inc, att_c)],
     C_k = mean of the co-kriging (+ FSI correction, which depends on the section through its
     stiffness), its standard deviation sigma(z) from the variances of the co-krigings, and the
     score mu + kappa sigma, mu = sum_c w_c dV_c (optimism under uncertainty: a section whose
     speed is uncertain is worth a look). The screened sections: a share local_fraction drawn in
     a trust region around the incumbent (radius shrunk when the incumbent is kept, widened when
     it changes: Conn, Gould & Toint 2000; TuRBO, Eriksson et al. 2019), the others anywhere in
     the family, with a seed of their own per run (screen_seed);
  2. the n_top best scores and the incumbent are solved with the full VPP (every condition);
     the best weighted speed becomes the incumbent;
  3. the integration points of the criterion are Sobol clouds around the solved VPP states of
     these sections (with a local search, half of them also move the section by the trust-region
     radius): the refinement goes where the decision "which section is best" is taken. Each point
     weighs w_c (dV/dC_k)_c^2 for coefficient k, c the condition of its state;
  4. every `sensitivities_every` updates, dV/dC_k at the incumbent by finite differences (with
     sensitivity_memory > 1, the median of the last evaluations: a path-dependent VPP solve
     doubled dV/dFx at one condition 3 times in the fourth run, 2026-10-27).
The first solve of a section that becomes the incumbent without a solved state (start of the
search, restart) also starts from GENERIC_GUESSES: from the fixed guess alone, the solve can stop
on a slow "successful" equilibrium (scripts/profile_analysis.robust_solve, 2026-10-26).
The candidates of a batch are new points in clouds around the states (vpp_optimize.candidates).
One batch = `batch_size` pairs (point, level), chosen one after the other by the best
speed-variance reduction per unit of cost (vpp_targeted.choose, the ALC gains of
sequential_design). After each choice the co-krigings are conditioned on the chosen point as if it
had been run, with its predicted mean as value ("kriging believer"): the variance of a GP does
not depend on the observed values, so the next choice sees exactly the variance reduction of the
previous ones, and the batch spreads by itself. Since 2026-10-27 this conditioning extends the
Cholesky factor (TorsorCoKriging.believe) instead of conditioning again on all the data: same
choices, 12 s less per choice at 3400 points. The codes of the batch then run in parallel
(CodeRunner), and the real values enter at the next fit. batch_size = 1 is the sequential loop of
the first two runs.

Convergence and restarts (2026-10-27; the fourth run gained 0.012 kt in its last 6 h 40). The
gain of an update is the weighted speed of the new incumbent minus that of the old one, both
solved in that update (same conditions). A change of less than min_gain counts as "kept" for the
trust-region radius. The region of the search has converged when its last convergence[0] updates
gained less than convergence[1] kt in all, or when its incumbent comes within merge_distance of a
local optimum already found (same basin). Then, up to max_restarts times, the incumbent is kept
as a local optimum and a new region starts from the best screened section of the whole family
at least restart_distance from every optimum (TuRBO restarts, Eriksson et al. 2019), all the data
kept; otherwise the loop stops. At the end, the optima and the last incumbent are solved again
with the final surrogate, and the best becomes the incumbent (compare_optima). The run also stops
at a wall-clock limit.

Parallel pieces (third run, 2026-10-22, 32 logical cores): the codes of a batch in a pool of
worker processes, the VPP solves of an update and the finite differences of the sensitivities in
another pool (VPPPool: one MDS per process), the 7 co-krigings in threads (TorsorCoKriging).

Costs of the criterion. Wall-clock per choice, overhead included. Sequential loop (first two
runs): level 1 ~0.2 s + overhead, level 2 (AVL) ~12 s + overhead, ratio 1 : 6. Batches with
NPLLT x XFOIL: the codes of a batch run at the same time, a choice costs the selection overhead
(~1-2 s, the same at both levels) and an HF run lengthens the batch only when its section is new
(XFOIL polar, 2-13 s; then 0.1 s per attitude): ratio 1 : 2.
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from src.data_management import greedy_maximin
from src.section_pca import section_ixx
from src.vpp_targeted import TorsorCoKriging, choose

ATTITUDE_BOUNDS = [(0.0, 8.0), (-1.5, 1.5), (-6.0, 6.0)]       # box of the C-foil RS
SPREAD = np.array([1.0, 0.3, 1.0])                              # cloud half-width [deg]
EXCLUSION_RADIUS = 1e-3            # a candidate this close to a failed run is not a candidate
OUTPUT_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Cpmin"]
# the worker processes compute one run at a time: one BLAS thread each
SINGLE_THREAD = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                        "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
# other starting points of the VPP solve (SOLVE_VARS order: speed, leeway, heel, trim, CC_Dspl,
# rudder, flat), besides its fixed initial guess (12 kt, flat 0.6): powered upwind, downwind
GENERIC_GUESSES = [np.array([18.0, 1.0, 5.0, 1.0, 9500.0, 0.5, 1.0]),
                   np.array([11.0, -2.0, 0.5, 0.0, 11000.0, 0.0, 1.0])]


def _process_pool(n_workers, initializer, initargs) -> ProcessPoolExecutor:
    """A pool whose workers run with one BLAS thread. The variables are read when a process
    imports numpy: the main process keeps the setting it started with, the workers (spawned on
    demand, at any time) read the new one."""
    os.environ.update(SINGLE_THREAD)
    return ProcessPoolExecutor(max_workers=n_workers, initializer=initializer, initargs=initargs)


# 1/5 ---------------------------------------------------------------------------------------------
# The problem
# -------------------------------------------------------------------------------------------------
class RealFoilProblem:
    """Sections of a family through a PCA, plus the 3 attitude angles of the foil RS.
    ixx_reference: Ixx of the current section; stiffness_ratios(scores) = ixx_reference / Ixx
    (1 if None), the scale of the FSI correction."""

    def __init__(self, pca, family, attitude_bounds=ATTITUDE_BOUNDS, spread=SPREAD,
                 ixx_reference=None):
        self.pca, self.family = pca, family
        self.n_scores = int(pca.n_modes)
        self.attitude_bounds = list(attitude_bounds)
        self.spread = np.asarray(spread, dtype=float)
        self.lo = np.array([b[0] for b in self.attitude_bounds])
        self.hi = np.array([b[1] for b in self.attitude_bounds])
        self.ixx_reference = ixx_reference
        self._bounds = None

    def sample_sections(self, n_sections: int, seed: int) -> np.ndarray:
        """Scores (n, k) of sections of the family whose DECODED section (the one the codes
        see: its projection on the k modes) still belongs to the family. With k = 10 and the Ixx
        constraint, the projection takes a little stiffness away: 24 % of the projected sections
        would leave the family (2026-10-22)."""
        kept, attempt = [], 0
        while sum(len(k) for k in kept) < n_sections:
            scores = self.pca.encode(self.family.sample(2 * n_sections, seed + 7919 * attempt))
            kept.append(scores[self.family.contains(self.pca.decode(scores))])
            attempt += 1
        return np.vstack(kept)[:n_sections]

    def weights(self, scores: np.ndarray) -> np.ndarray:
        return self.pca.decode(np.atleast_2d(scores))

    def stiffness_ratios(self, scores: np.ndarray) -> np.ndarray:
        scores = np.atleast_2d(scores)
        if self.ixx_reference is None:
            return np.ones(len(scores))
        return self.ixx_reference / section_ixx(self.weights(scores), self.family.te_thickness)

    def bounds(self) -> list:
        if self._bounds is None:
            scores = self.sample_sections(4096, seed=101)
            self._bounds = [(float(a), float(b)) for a, b in zip(scores.min(0), scores.max(0))] \
                + self.attitude_bounds
        return self._bounds

    def local_sections(self, center: np.ndarray, radius: float, n_sections: int, seed: int,
                       max_attempts: int = 20) -> np.ndarray:
        """
        Scores (<= n, k) of sections around `center`, the trust region of the local search
        (VPPSectionOptimizer.screen): center + radius g / sqrt(k), g ~ N(0, I), i.e. a typical
        distance `radius` in the whitened scores (1 = one standard deviation of the family),
        kept when the decoded section belongs to the family.
        """
        rng = np.random.default_rng(seed)
        kept, total = [], 0
        for _ in range(max_attempts):
            scores = center + radius * rng.standard_normal((2 * n_sections, self.n_scores)) \
                / np.sqrt(self.n_scores)
            kept.append(scores[self.family.contains(self.weights(scores))])
            total += len(kept[-1])
            if total >= n_sections:
                break
        return np.vstack(kept)[:n_sections]

    def cloud(self, points: np.ndarray, n_per_state: int, seed: int,
              section_radius: float = 0.0) -> tuple:
        """
        Cloud of n_per_state points around each point (a VPP state): attitude +- spread (Sobol,
        clipped to the box). With section_radius > 0, every second point also moves the section,
        by an offset of typical length section_radius in the whitened scores (as local_sections),
        and the points whose decoded section leaves the family are dropped: half of the cloud
        serves the competition between the sections of the states, half their neighbourhood
        (the local search). Returns the points (m, d) and the index of the state of each (m,).
        """
        points = np.atleast_2d(points)
        sobol = 2.0 * qmc.Sobol(d=len(self.spread), scramble=True, seed=seed).random(n_per_state) - 1.0
        rng = np.random.default_rng(seed)
        out, owner = [], []
        for i, point in enumerate(points):
            attitude = np.clip(point[self.n_scores:] + sobol * self.spread, self.lo, self.hi)
            scores = np.repeat(point[None, :self.n_scores], n_per_state, 0)
            if section_radius > 0.0:
                scores[1::2] += section_radius * rng.standard_normal(scores[1::2].shape) \
                    / np.sqrt(self.n_scores)
            out.append(np.column_stack([scores, attitude]))
            owner.append(np.full(n_per_state, i))
        x, owner = np.vstack(out), np.concatenate(owner)
        if section_radius > 0.0:
            keep = self.family.contains(self.weights(x[:, :self.n_scores]))
            x, owner = x[keep], owner[keep]
        return x, owner


# 2/5 ---------------------------------------------------------------------------------------------
# The codes, in parallel worker processes
# -------------------------------------------------------------------------------------------------
_WORKER = {}


def _init_worker(work_dir, levels):
    from src.foil_codes import FoilCodes  # noqa: PLC0415
    _WORKER["codes"] = FoilCodes(work_dir, levels=levels)


def _run_one(task):
    weights, attitude, level = task
    y, info = _WORKER["codes"].evaluate(weights, attitude, level)
    return y, info


class CodeRunner:
    """
    Runs the codes on points of the problem and records every run in the run folder.
    `levels`: the levels of foil_codes (LEVELS or LEVELS_XFOIL); a run of level l uses
    levels[l - 1]. Several points go to a pool of worker processes, kept for the whole run; a
    single point runs in the main process.
    """

    def __init__(self, problem: RealFoilProblem, run, n_workers: int = 6, levels=None):
        from src.foil_codes import LEVELS, FoilCodes  # noqa: PLC0415
        self.problem, self.run, self.n_workers = problem, run, int(n_workers)
        self.levels = list(levels or LEVELS)
        self.work_dir = run.path("data", "codes")
        self.local = FoilCodes(self.work_dir, levels=self.levels)
        self.costs = self.local.costs
        self.n_runs = {level: 0 for level in range(1, len(self.levels) + 1)}
        self.time = {level: 0.0 for level in range(1, len(self.levels) + 1)}
        self.pool = None

    def run_tasks(self, points: np.ndarray, levels, tag: str = "") -> tuple:
        """One run per point, at its level: outputs (m, 7) (NaN rows = failed runs) and the
        infos, in the order of the points."""
        points = np.atleast_2d(points)
        levels = np.broadcast_to(np.asarray(levels, dtype=int), (len(points),))
        weights = self.problem.weights(points[:, :self.problem.n_scores])
        tasks = [(w, p[self.problem.n_scores:], int(level))
                 for w, p, level in zip(weights, points, levels)]
        if len(tasks) == 1 or self.n_workers <= 1:
            results = [self.local.evaluate(*task) for task in tasks]
        else:
            if self.pool is None:
                self.pool = _process_pool(self.n_workers, _init_worker,
                                          (str(self.work_dir), self.levels))
            results = list(self.pool.map(_run_one, tasks))
        y = np.array([r[0] for r in results])
        for point, level, (values, info) in zip(points, levels, results):
            self.n_runs[int(level)] += 1
            self.time[int(level)] += info["time_s"]
            self.run.add_evaluation({
                "tag": tag, "level": int(level), "ok": not info["error"],
                **{f"x{i}": float(v) for i, v in enumerate(point)},
                **dict(zip(OUTPUT_NAMES, (float(v) for v in values))),
                "time_s": round(info["time_s"], 3), "error": info["error"][:200],
                "elapsed_min": round(self.run.elapsed() / 60, 3)})
        return y, [r[1] for r in results]

    def evaluate(self, points: np.ndarray, level: int, tag: str = "") -> tuple:
        """All the points at one level (start design, validation)."""
        return self.run_tasks(points, level, tag)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


def start_design(problem: RealFoilProblem, runner: CodeRunner, n_low: int, n_high: int,
                 seed: int, n_models: int = 7, costs=(1.0, 6.0),
                 n_threads: int = 1) -> TorsorCoKriging:
    """Nested design: sections of the family, attitude by Latin hypercube, HF by maximin among
    the LF points that succeeded; failed runs are left out (recorded in the run folder)."""
    scores = problem.sample_sections(n_low, seed)
    attitude = problem.lo + qmc.LatinHypercube(d=len(problem.lo), seed=seed).random(n_low) \
        * (problem.hi - problem.lo)
    x_low = np.column_stack([scores, attitude])
    y_low, _ = runner.evaluate(x_low, 1, tag="start")
    ok = np.all(np.isfinite(y_low), axis=1)
    x_low, y_low = x_low[ok], y_low[ok]
    x_high = greedy_maximin(x_low, n_high, problem.bounds())
    y_high, _ = runner.evaluate(x_high, 2, tag="start")
    ok_high = np.all(np.isfinite(y_high), axis=1)
    surrogate = TorsorCoKriging(problem.bounds(), costs=costs, seed=seed, n_models=n_models,
                                n_threads=n_threads)
    surrogate.add(1, x_low, y_low)
    surrogate.add(2, x_high[ok_high], y_high[ok_high])
    runner.run.log(f"start design: LF {len(x_low)} ok / {n_low}, HF {ok_high.sum()} ok / "
                   f"{len(x_high)}")
    return surrogate


# 3/5 ---------------------------------------------------------------------------------------------
# The VPP solves, in parallel worker processes
# -------------------------------------------------------------------------------------------------
_VPP_WORKER = {}


def _init_vpp_worker(options):
    from src.vpp_coupling import FSICorrection, IntensSYVPP  # noqa: PLC0415
    fsi = FSICorrection() if options["fsi"] else None
    _VPP_WORKER["vpp"] = IntensSYVPP(foil=options["foil"], with_hvol=options["with_hvol"], fsi=fsi)
    _VPP_WORKER["version"] = None


def _vpp_task(task):
    """("solve", ...) -> state dict; ("sensitivity", ...) -> dV/dC_k. For a solve, base_state
    carries the extra starting points (IntensSYVPP.solve keeps the fastest converged state)."""
    kind, path, version, scores, ratio, twa, tws, guess, base_state, k = task
    if _VPP_WORKER["version"] != version:
        with open(path, "rb") as file:
            _VPP_WORKER["surrogate"] = pickle.load(file)
        _VPP_WORKER["version"] = version
    vpp, predict = _VPP_WORKER["vpp"], _VPP_WORKER["surrogate"].predict
    vpp.set_foil(predict, scores, ratio)
    if kind == "solve":
        return vpp.solve(twa, tws, guess=guess, extra_guesses=base_state or ())
    # the base state is solved again in THIS process: the MDS state carries over from one solve
    # to the next (path dependence), so the finite difference must take two solves of the same
    # process (with the base state of another worker, the smoke test of 2026-10-22 gave dV/dFx
    # of -32 instead of +83)
    base = vpp.solve(twa, tws, guess=base_state["unknowns"])
    if not base["success"]:
        return float("nan")
    return float(vpp.speed_sensitivities(predict, scores, twa, tws, base,
                                         stiffness_ratio=ratio, outputs=[k])[k])


class VPPPool:
    """
    The VPP solves of an update (sections x conditions) and the finite differences of the
    sensitivities (conditions x 6 coefficients), in parallel: one IntensSYVPP (MDS) per worker
    process. The surrogate goes to the workers through a pickle file, written once per new
    version (publish), read once per worker.
    Each worker has its own MDS state: the path dependence of the solve (vpp_coupling, SLSQP)
    differs from one worker to another, as it differed from one call to the next in the
    sequential loop.
    """

    def __init__(self, run, n_workers: int, options: dict):
        self.run, self.path, self.version = run, None, 0
        self.pool = _process_pool(n_workers, _init_vpp_worker, (options,))

    def publish(self, surrogate) -> None:
        """The means of the surrogate for the workers (TorsorCoKriging.mean_only_copy)."""
        old = self.path
        self.version += 1
        self.path = self.run.path("data", f"surrogate_v{self.version}.pkl")
        with open(self.path, "wb") as file:
            pickle.dump(surrogate.mean_only_copy(), file, protocol=pickle.HIGHEST_PROTOCOL)
        if old is not None and old.exists():
            old.unlink()                  # the workers have read it (the maps are synchronous)

    def solve(self, sections, ratios, conditions, guesses=None, extra_guesses=None) -> list:
        """States [section][condition] (vpp_coupling.IntensSYVPP.solve). guesses: one starting
        point per condition (None = the fixed initial guess of the VPP); extra_guesses: None or
        [section][condition] lists of other starting points, the fastest converged state is kept
        (the solve is path dependent: profile_analysis, 2026-10-26)."""
        tasks = [("solve", self.path, self.version, scores, float(ratio), twa, tws,
                  None if guesses is None else guesses[j],
                  None if extra_guesses is None else list(extra_guesses[i][j]), None)
                 for i, (scores, ratio) in enumerate(zip(sections, ratios))
                 for j, (twa, tws) in enumerate(conditions)]
        states = list(self.pool.map(_vpp_task, tasks))
        n = len(conditions)
        return [states[i * n:(i + 1) * n] for i in range(len(sections))]

    def sensitivities(self, scores, ratio, conditions, base_states, n_outputs=6) -> np.ndarray:
        """dV/dC_k (n_conditions, n_outputs) at the section, from its base states."""
        tasks = [("sensitivity", self.path, self.version, scores, float(ratio), twa, tws, None,
                  state, k)
                 for (twa, tws), state in zip(conditions, base_states) for k in range(n_outputs)]
        return np.array(list(self.pool.map(_vpp_task, tasks))).reshape(len(conditions), n_outputs)

    def close(self) -> None:
        self.pool.shutdown()


# 4/5 ---------------------------------------------------------------------------------------------
# The optimizer
# -------------------------------------------------------------------------------------------------
class VPPSectionOptimizer:
    """The loop of the module docstring. vpp_pool: None (VPP solves in the main process, on
    `vpp`) or a VPPPool; `vpp` is used for its FSI correction in any case."""

    def __init__(self, problem, surrogate, runner, vpp, conditions, run, time_limit_s,
                 n_per_state=16, n_local=512, refit_every=10, update_every=10,
                 sensitivities_every=2, n_screen=256, n_top=3, kappa=1.0, seed=0,
                 start_scores=None, batch_size=1, vpp_pool=None, refit_growth=None,
                 condition_weights=None, screen_seed=None, local_fraction=0.0, radius=1.0,
                 radius_bounds=(0.1, 3.0), radius_factors=(0.7, 1.5), max_refit_size=None,
                 min_gain=0.0, convergence=None, max_restarts=0, restart_distance=2.0,
                 merge_distance=1.0, sensitivity_memory=1):
        """
        - refit_growth: new hyperparameters when the design has grown by this factor since the
          last fit (e.g. 1.25; a fit costs O(n^3) and the batches add many points), instead of
          every refit_every batches.
        - condition_weights: probability of each condition (the nodes of an operating profile,
          src/operating_profile.py); None = the same weight for every condition.
        - screen_seed: seed of the sections drawn by the screening (None = seed); a new one per
          run, otherwise two runs screen the same sections (runs 2 and 3, 2026-10-22).
        - local_fraction: share of the screened sections drawn around the incumbent, in a trust
          region of typical radius `radius` (whitened scores); 0 = no local search. The radius is
          multiplied by radius_factors[0] when the incumbent is kept, by radius_factors[1] when
          it changes, within radius_bounds (trust-region rule). With a local search, the clouds of
          the criterion also move the section by this radius (RealFoilProblem.cloud).
        - max_refit_size: above this design size the hyperparameters are frozen (conditioning
          only): a fit costs O(n^3), ~2.5 min at 1045 points in the third run, about an hour at
          3000 (None = no limit).
        - min_gain [kt]: a change of incumbent that gains less counts as "kept" for the radius.
        - convergence: (window, tolerance [kt]) of the convergence test, None = no test (runs 1
          to 4: until the time limit); max_restarts, restart_distance, merge_distance (whitened
          scores): the restarts (module docstring).
        - sensitivity_memory: the sensitivities are the median of that many last evaluations
          (1 = the last one).
        """
        self.problem, self.surrogate, self.runner, self.vpp = problem, surrogate, runner, vpp
        self.conditions, self.run = list(conditions), run
        weights = np.ones(len(self.conditions)) if condition_weights is None \
            else np.asarray(condition_weights, dtype=float)
        self.condition_weights = weights / weights.sum()
        self.time_limit_s = float(time_limit_s)
        self.n_per_state, self.n_local = n_per_state, n_local
        self.refit_every, self.update_every = refit_every, update_every
        self.sensitivities_every = sensitivities_every
        self.n_screen, self.n_top, self.kappa, self.seed = n_screen, n_top, kappa, seed
        self.screen_seed = seed if screen_seed is None else int(screen_seed)
        self.local_fraction, self.radius = float(local_fraction), float(radius)
        self.radius_bounds, self.radius_factors = radius_bounds, radius_factors
        self.batch_size, self.vpp_pool = int(batch_size), vpp_pool
        self.refit_growth, self.max_refit_size = refit_growth, max_refit_size
        self.fitted_size = None                            # design size at the last refit
        self.iteration, self.n_batches, self.n_updates = 0, 0, 0
        # start of the search: the given section (e.g. the current C-foil section), or the mean
        # section of the PCA
        self.incumbent = np.zeros(problem.n_scores) if start_scores is None \
            else np.asarray(start_scores, dtype=float)
        self.incumbent_states = None
        self.derivatives = None                            # (n_conditions, 6)
        # the solved states of the last update (m, d), the condition of each (m,), the integration
        # points and the state of each
        self.states, self.state_conditions = None, None
        self.x_int, self.int_owner = None, None
        self.history, self.updates = [], []
        # points where a run failed: excluded from the candidates (the first real run chose the
        # same failing HF point 4 times in a row, 40 s each, 2026-10-22)
        self.failed = []
        # convergence and restarts (module docstring): the gains of the updates of the current
        # region, the local optima of the converged regions
        self.min_gain, self.convergence = float(min_gain), convergence
        self.max_restarts, self.restart_distance = int(max_restarts), float(restart_distance)
        self.merge_distance, self.initial_radius = float(merge_distance), float(radius)
        self.region, self.region_gains, self.optima, self.compared = 0, [], [], []
        self.new_region = True                             # sensitivities due at the next update
        self.sensitivity_memory, self.sensitivity_history = int(sensitivity_memory), []

    # --- VPP pieces ---
    def solve_sections(self, sections, guesses=None, extra_guesses=None) -> list:
        """States [section][condition] of several sections, with the current surrogate
        (guesses, extra_guesses: VPPPool.solve)."""
        ratios = self.problem.stiffness_ratios(np.array(sections))
        if self.vpp_pool is not None:                # the surrogate published by update()
            return self.vpp_pool.solve(sections, ratios, self.conditions, guesses, extra_guesses)
        out = []
        for i, (scores, ratio) in enumerate(zip(sections, ratios)):
            self.vpp.set_foil(self.surrogate.predict, scores, ratio)
            out.append([self.vpp.solve(twa, tws, guess=None if guesses is None else guesses[j],
                                       extra_guesses=() if extra_guesses is None
                                       else extra_guesses[i][j])
                        for j, (twa, tws) in enumerate(self.conditions)])
        return out

    def solve_section(self, scores, guesses=None, extra_guesses=None) -> list:
        return self.solve_sections([scores], guesses,
                                   None if extra_guesses is None else [extra_guesses])[0]

    def state_points(self, scores, states) -> np.ndarray:
        return np.array([np.concatenate([scores, [s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]]])
                         for s in states])

    def sensitivities(self) -> None:
        """dV/dC_k at the solved states of the incumbent (the other conditions keep their
        previous values: a state without equilibrium has no sensitivity)."""
        ratio = float(self.problem.stiffness_ratios(self.incumbent)[0])
        solved = [j for j, s in enumerate(self.incumbent_states) if s["success"]]
        conditions = [self.conditions[j] for j in solved]
        states = [self.incumbent_states[j] for j in solved]
        derivatives = np.full((len(self.conditions), 6), np.nan)
        if solved and self.vpp_pool is not None:
            derivatives[solved] = self.vpp_pool.sensitivities(self.incumbent, ratio, conditions,
                                                              states)
        elif solved:
            derivatives[solved] = np.array([
                self.vpp.speed_sensitivities(self.surrogate.predict, self.incumbent, twa, tws,
                                             state, stiffness_ratio=ratio)
                for (twa, tws), state in zip(conditions, states)])
        bad = ~np.isfinite(derivatives)
        if bad.any():
            # a failed solve: keep the previous value of that derivative (0 the first time)
            previous = self.derivatives if self.derivatives is not None \
                else np.zeros_like(derivatives)
            derivatives[bad] = previous[bad]
            self.run.log(f"speed sensitivities: {bad.sum()} failed solves, previous values kept")
        # the median of the last evaluations at this incumbent's region (module docstring)
        self.sensitivity_history.append(derivatives)
        self.derivatives = np.median(self.sensitivity_history[-self.sensitivity_memory:], axis=0)
        outliers = np.abs(derivatives - self.derivatives) > 0.3 * np.abs(self.derivatives)
        if self.sensitivity_memory > 1 and outliers.any():
            self.run.log(f"speed sensitivities: {outliers.sum()} values more than 30 % from the "
                         f"median of the last {self.sensitivity_memory} evaluations, median kept")
        self.run.log("speed sensitivities dV/dC (rows = conditions): "
                     + np.array2string(self.derivatives, precision=2, suppress_small=True))

    def screen(self) -> np.ndarray:
        """
        Scores (n_top, k) of the best sections by mu + kappa sigma, over the solved states c of
        the incumbent, with the probabilities w_c of the conditions:
            mu = sum_c w_c dV_c / sum_c w_c,   sigma^2 = sum_c w_c^2 Var V_c / (sum_c w_c)^2.
        The screened sections: local_fraction of them around the incumbent (trust region of
        radius self.radius), the others anywhere in the family.
        """
        seed = self.screen_seed + 31 * self.n_updates
        n_local = int(round(self.local_fraction * self.n_screen))
        parts = [self.incumbent[None]]
        if n_local:
            parts.append(self.problem.local_sections(self.incumbent, self.radius, n_local,
                                                     seed + 1))
        parts.append(self.problem.sample_sections(self.n_screen - n_local, seed))
        candidates = np.vstack(parts)
        gain, sigma = self.first_order_scores(candidates)
        score = gain + self.kappa * sigma
        order = np.argsort(-score[1:])[:self.n_top] + 1
        self.last_screen = {"gain": gain[order].tolist(), "sigma": sigma[order].tolist(),
                            "incumbent_sigma": float(sigma[0]), "n_local": len(parts[1])
                            if n_local else 0, "local_in_top": int(np.sum(order <= len(parts[1])))
                            if n_local else 0}
        return candidates[order]

    def first_order_scores(self, candidates: np.ndarray) -> tuple:
        """mu and sigma of the screening (screen) for sections (m, k) whose first row is the
        incumbent (its mu is 0)."""
        ratios = self.problem.stiffness_ratios(candidates)
        gain = np.zeros(len(candidates))
        variance = np.zeros(len(candidates))
        total = 0.0
        for j, state in enumerate(self.incumbent_states):
            if not state["success"]:
                continue
            w = self.condition_weights[j]
            total += w
            attitude = np.array([state["Cant_Eff"], state["Rake_Eff"], state["Yaw_Eff"]])
            x = np.column_stack([candidates, np.repeat(attitude[None], len(candidates), 0)])
            means = self.surrogate.predict(x)[:, :6]
            gain += w * (means - means[0]) @ self.derivatives[j]
            if self.vpp.fsi is not None:
                # the FSI correction of a section = its stiffness ratio x the one of the RS
                delta = self.vpp.fsi(attitude[None], state["speed"])[0, :6]
                gain += w * (ratios - ratios[0]) * (delta @ self.derivatives[j])
            variance += w ** 2 * self.surrogate.variances(x)[:, :6] @ self.derivatives[j] ** 2
        total = total if total > 0.0 else 1.0           # no solved state: nothing to rank on
        return gain / total, np.sqrt(variance) / total

    def update(self) -> None:
        """Screening, full VPP of the best candidates and of the incumbent, new integration
        points, new sensitivities (module docstring)."""
        start = time.perf_counter()
        if self.vpp_pool is not None:
            self.vpp_pool.publish(self.surrogate)    # the model of this update, for the workers
        if self.incumbent_states is None:
            # several starting points: from the fixed guess alone the solve can stop on a slow
            # "successful" equilibrium (module docstring)
            self.incumbent_states = self.solve_section(
                self.incumbent, extra_guesses=[GENERIC_GUESSES] * len(self.conditions))
        if self.derivatives is None or self.new_region \
                or self.n_updates % self.sensitivities_every == 0:
            self.sensitivities()
            self.new_region = False
        top = self.screen()
        sections = [self.incumbent] + list(top)
        guesses = [s["unknowns"] for s in self.incumbent_states]
        all_states = self.solve_sections(sections, guesses)
        # the speed of a section: probability-weighted mean over the conditions solved for ALL
        # the sections of the update, so that they are compared on the same conditions
        solved = np.array([[s["success"] for s in states] for states in all_states])
        weights = self.condition_weights * solved.all(axis=0)
        results = []
        for scores, states in zip(sections, all_states):
            speeds = np.array([s["speed"] for s in states])
            mean = float(np.sum(weights * speeds) / weights.sum()) if weights.sum() > 0 else np.nan
            results.append((mean, scores, states,
                            [s["speed"] if s["success"] else np.nan for s in states]))
        best = max(results, key=lambda r: r[0] if np.isfinite(r[0]) else -np.inf)
        changed = best[1] is not self.incumbent and not np.allclose(best[1], self.incumbent)
        # gain of the update: new incumbent against the old one (results[0]), same conditions;
        # the speed itself is not comparable from one update to the next (other model, other
        # conditions solved for all)
        gain = best[0] - results[0][0] if changed else 0.0
        gain = float(gain) if np.isfinite(gain) else 0.0
        self.region_gains.append(gain)
        self.incumbent, self.incumbent_states = best[1], best[2]
        if self.local_fraction > 0.0:
            # trust region of the local search: wider when the incumbent changes by at least
            # min_gain, narrower otherwise
            low, high = self.radius_bounds
            shrink, grow = self.radius_factors
            self.radius = min(high, self.radius * grow) if changed and gain >= self.min_gain \
                else max(low, self.radius * shrink)
        # the solved states of the 4 sections, the condition of each, the integration points
        rows = [(self.state_points(r[1], [s])[0], j) for r in results
                for j, s in enumerate(r[2]) if s["success"]]
        self.states = np.array([row for row, _ in rows])
        self.state_conditions = np.array([j for _, j in rows])
        self.x_int, self.int_owner = self.problem.cloud(
            self.states, self.n_per_state, self.seed + self.iteration, self.section_radius())
        record = {"update": self.n_updates, "iteration": self.iteration,
                  "elapsed_min": self.run.elapsed() / 60, "cost": self.surrogate.cost,
                  "n_lf": self.surrogate.sizes[0], "n_hf": self.surrogate.sizes[1],
                  "incumbent_speed": float(best[0]), "incumbent_speeds": best[3],
                  "probability_compared": float(weights.sum()), "radius": self.radius,
                  "changed": bool(changed), "gain": gain, "region": self.region,
                  "candidate_speeds": [float(r[0]) for r in results],
                  "incumbent": self.incumbent.tolist(), **self.last_screen,
                  "time_s": time.perf_counter() - start}
        self.updates.append(record)
        self.run.log(f"update {self.n_updates}: incumbent speed {best[0]:.4f} kt "
                     f"({'new' if changed else 'kept'}, gain {gain:.4f} kt), candidates "
                     f"{np.round(record['candidate_speeds'], 3)}, on "
                     f"{100 * weights.sum():.0f} % of the probability, radius {self.radius:.2f}, "
                     f"region {self.region}, {record['time_s']:.0f} s")
        self.n_updates += 1

    def section_radius(self) -> float:
        """How far the clouds of the criterion move the section: the trust-region radius with
        a local search, 0 (section of the state only) without."""
        return self.radius if self.local_fraction > 0.0 else 0.0

    def point_weights(self) -> np.ndarray:
        """
        (N, 6) weight of every integration point for every coefficient:
            w_c x (dV/dC_k)_c^2 / (number of points of its state),
        c the condition of the state of the point: a state counts for the probability of its
        condition, whatever the number of its points kept (vpp_targeted.choose).
        """
        conditions = self.state_conditions[self.int_owner]
        counts = np.bincount(self.int_owner, minlength=len(self.states))[self.int_owner]
        return (self.condition_weights[conditions] / counts)[:, None] \
            * self.derivatives[conditions] ** 2

    # --- convergence and restarts (module docstring) ---
    def converged(self) -> bool:
        """The region of the search has converged: its last convergence[0] updates gained less
        than convergence[1] kt in all, or its incumbent is in the basin of an optimum already
        found (closer than merge_distance)."""
        if self.convergence is None:
            return False
        window, tolerance = self.convergence
        if len(self.region_gains) >= window and sum(self.region_gains[-window:]) < tolerance:
            return True
        return any(np.linalg.norm(self.incumbent - optimum["scores"]) < self.merge_distance
                   for optimum in self.optima)

    def restart(self) -> None:
        """Keeps the incumbent as a local optimum and starts a new region from the best screened
        section of the whole family (mu + kappa sigma, as screen) at least restart_distance from
        every optimum; its states are solved at the next update."""
        last = self.updates[-1]
        self.optima.append({"scores": self.incumbent.copy(), "states": self.incumbent_states,
                            "region": self.region, "speed": last["incumbent_speed"],
                            "update": last["update"], "elapsed_min": last["elapsed_min"]})
        pool = self.problem.sample_sections(4 * self.n_screen,
                                            self.screen_seed + 31 * self.n_updates + 17)
        distances = np.min([np.linalg.norm(pool - optimum["scores"], axis=1)
                            for optimum in self.optima], axis=0)
        far = pool[distances >= self.restart_distance]
        if not len(far):
            far = pool[[np.argmax(distances)]]               # the farthest one, failing better
        gain, sigma = self.first_order_scores(np.vstack([self.incumbent[None], far]))
        best = int(np.argmax(gain[1:] + self.kappa * sigma[1:]))
        distance = min(np.linalg.norm(far[best] - optimum["scores"]) for optimum in self.optima)
        self.region += 1
        self.incumbent, self.incumbent_states = far[best], None
        self.radius, self.region_gains = self.initial_radius, []
        self.new_region, self.sensitivity_history = True, []
        self.run.log(f"region {self.region - 1} converged (update {last['update']}, incumbent "
                     f"speed {last['incumbent_speed']:.4f} kt): region {self.region} starts from a "
                     f"section at distance {distance:.2f} of the optima, predicted gain "
                     f"{gain[1 + best]:+.3f} +- {sigma[1 + best]:.3f} kt")

    def compare_optima(self) -> None:
        """The local optima of the regions and the incumbent, solved again with the current
        surrogate from every state known at each condition and GENERIC_GUESSES; the best
        weighted speed over the conditions solved for all becomes the incumbent."""
        if self.vpp_pool is not None:
            self.vpp_pool.publish(self.surrogate)
        entries = self.optima + [{"scores": self.incumbent, "states": self.incumbent_states,
                                  "region": self.region}]
        known = [[e["states"][j]["unknowns"] for e in entries if e["states"][j]["success"]]
                 + GENERIC_GUESSES for j in range(len(self.conditions))]
        all_states = self.solve_sections([e["scores"] for e in entries],
                                         extra_guesses=[known] * len(entries))
        solved = np.array([[s["success"] for s in states] for states in all_states])
        weights = self.condition_weights * solved.all(axis=0)
        speeds = [float(np.sum(weights * np.array([s["speed"] for s in states])) / weights.sum())
                  if weights.sum() > 0 else np.nan for states in all_states]
        if not np.isfinite(speeds).any():
            self.run.log("local optima: no condition solved for all of them, incumbent kept")
            return
        best = int(np.nanargmax(speeds))
        self.compared = [{"region": e["region"], "scores": np.asarray(e["scores"]).tolist(),
                          "speed_when_found": e.get("speed"), "final_speed": s,
                          "final_speeds": [st["speed"] if st["success"] else None for st in states],
                          "best": i == best}
                         for i, (e, s, states) in enumerate(zip(entries, speeds, all_states))]
        self.run.log("local optima with the final surrogate, on "
                     f"{100 * weights.sum():.0f} % of the probability: "
                     + ", ".join(f"region {e['region']} {s:.4f} kt" for e, s in zip(entries, speeds))
                     + f" -> region {entries[best]['region']}")
        if best != len(entries) - 1:
            self.incumbent, self.incumbent_states = entries[best]["scores"], all_states[best]
            self.new_region, self.sensitivity_history = True, []
        else:
            self.incumbent_states = all_states[best]

    # --- one batch and the loop ---
    def candidates(self) -> tuple:
        """Candidates (M, d) and their missing levels (M, 2): new points in clouds around the
        states (n_local in all; the section moves too with a local search), minus the neighbours
        of failed runs. The points far from the states and the LF points without HF of earlier
        batches are no longer candidates: in the third run, 100 % of the choices were near the
        states and 2 HF out of 252 were on an earlier LF point (2026-10-23)."""
        x_cand, _ = self.problem.cloud(self.states, max(1, self.n_local // len(self.states)),
                                       self.seed + 7 * self.iteration + 1, self.section_radius())
        missing = np.ones((len(x_cand), 2), dtype=bool)
        if self.failed:
            # distance in the unit cube of the bounds, as sequential_design.EXCLUSION_RADIUS
            span = np.array([b[1] - b[0] for b in self.problem.bounds()])
            gaps = np.abs(x_cand[:, None, :] - np.array(self.failed)[None]) / span
            keep = np.min(np.max(gaps, axis=2), axis=1) > EXCLUSION_RADIUS
            x_cand, missing = x_cand[keep], missing[keep]
        return x_cand, missing

    @staticmethod
    def levels_to_run(choice) -> list:
        """The missing levels up to the chosen one (nested design)."""
        return [level for level in (1, 2)
                if level <= choice["level"] and choice["missing"][level - 1]]

    def step(self) -> list:
        """One batch (module docstring): choices, codes in parallel, data. Returns the records."""
        start = time.perf_counter()
        size = sum(self.surrogate.sizes)
        if self.refit_growth:
            refit = self.fitted_size is None or size >= self.refit_growth * self.fitted_size
        else:
            refit = self.n_batches % self.refit_every == 0
        if refit and self.fitted_size is not None and not self.refit_allowed(size):
            refit = False                                  # hyperparameters frozen (docstring)
        self.surrogate.fit(refit=refit)
        if refit:
            self.fitted_size = size
            self.run.log(f"batch {self.n_batches}: hyperparameters refitted on {size} points")
        if self.converged():
            if len(self.optima) >= self.max_restarts:
                self.run.log(f"region {self.region} converged, no restart left: end of the loop")
                return []
            self.restart()
        update_start = time.perf_counter()
        if self.n_batches % self.update_every == 0 or self.states is None \
                or self.incumbent_states is None:
            self.update()
        update_s = time.perf_counter() - update_start
        weights = self.point_weights()                     # (N, 6): Cpmin weight 0
        x_cand, missing = self.candidates()
        choices = []
        for _ in range(self.batch_size):
            choice = choose(self.surrogate, weights, x_cand, missing, self.x_int, np.inf)
            if choice["level"] is None:
                break
            choices.append(choice)
            missing[choice["index"], :choice["level"]] = False
            if len(choices) < self.batch_size:          # kriging believer (module docstring)
                self.surrogate.believe(choice["x"], self.levels_to_run(choice))
        select_s = time.perf_counter() - start

        tasks = [(i, level) for i, choice in enumerate(choices) for level in self.levels_to_run(choice)]
        y, infos = self.runner.run_tasks(np.array([choices[i]["x"] for i, _ in tasks]),
                                         [level for _, level in tasks], tag=f"batch{self.n_batches}")
        results = {task: (y[row], infos[row]) for row, task in enumerate(tasks)}
        records = []
        for i, choice in enumerate(choices):
            x, ok = choice["x"], True
            for level in self.levels_to_run(choice):
                values, info = results[(i, level)]
                in_lf = level == 1 or np.any(np.all(np.isclose(self.surrogate.x_lists[0], x), axis=1))
                if not np.all(np.isfinite(values)) or not in_lf:
                    ok = False
                    self.failed.append(x.copy())
                    self.run.log(f"batch {self.n_batches}: level {level} failed "
                                 f"({info['error'][:80] or 'its level 1 failed'})")
                    break
                self.surrogate.add(level, x, values)
            distances = np.linalg.norm((self.states[:, self.problem.n_scores:]
                                        - x[self.problem.n_scores:]) / self.problem.spread, axis=1)
            same_section = np.all(np.isclose(self.states[:, :self.problem.n_scores],
                                             x[:self.problem.n_scores]), axis=1)
            records.append({"iteration": self.iteration, "batch": self.n_batches,
                            "elapsed_min": self.run.elapsed() / 60, "level": choice["level"],
                            "ratio": choice["ratio"], "ok": ok,
                            "near_state": bool(same_section.any()),
                            "attitude_distance": float(distances[same_section].min())
                            if same_section.any() else float("nan"),
                            "x": x.tolist(), "time_s": (time.perf_counter() - start) / len(choices),
                            "select_s": select_s, "update_s": update_s,
                            "n_lf": self.surrogate.sizes[0], "n_hf": self.surrogate.sizes[1],
                            "region": self.region})
            self.iteration += 1
        self.history.extend(records)
        self.n_batches += 1
        return records

    def run_loop(self) -> None:
        while self.run.elapsed() < self.time_limit_s:
            records = self.step()
            if not records:
                break
            if self.batch_size > 1 or self.iteration % 5 == 0:
                last = records[-1]
                n_high = sum(r["level"] == 2 for r in records)
                self.run.log(f"batch {last['batch']}: {len(records)} choices ({n_high} HF), LF/HF "
                             f"{last['n_lf']}/{last['n_hf']}, selection {last['select_s']:.1f} s "
                             f"(update {last['update_s']:.0f} s), "
                             f"{last['time_s'] * len(records):.1f} s")
        self.surrogate.fit(refit=self.refit_allowed(sum(self.surrogate.sizes)))
        if self.optima:
            self.compare_optima()                          # the best region, final model
        self.update()                                      # final incumbent with the final model

    def refit_allowed(self, size: int) -> bool:
        return self.max_refit_size is None or size <= self.max_refit_size


# 5/5 ---------------------------------------------------------------------------------------------
# Validation of the final surrogate with the codes at the VPP states
# -------------------------------------------------------------------------------------------------
def validate(optimizer: VPPSectionOptimizer, sections: dict, extra_levels=None) -> list:
    """The codes at the VPP states of the given sections (name -> scores): HF (level 2, the
    reference of the surrogate) and LF, plus extra_levels {name: level} (e.g. {"avl": 3});
    surrogate mean and sigma against HF, and the first-order speed error per state."""
    rows = []
    names = list(sections)
    guesses = [s["unknowns"] for s in optimizer.incumbent_states]
    all_states = optimizer.solve_sections([sections[n] for n in names], guesses)
    for name, states in zip(names, all_states):
        scores = sections[name]
        points = optimizer.state_points(scores, states)
        codes = {"hf": 2, "lf": 1, **(extra_levels or {})}
        values = {key: optimizer.runner.evaluate(points, level, tag=f"validation_{name}")[0]
                  for key, level in codes.items()}
        mean = optimizer.surrogate.predict(points)
        sigma = np.sqrt(optimizer.surrogate.variances(points))
        for j, (state, point) in enumerate(zip(states, points)):
            error = mean[j, :6] - values["hf"][j, :6]
            speed_error = float(error @ optimizer.derivatives[j])
            speed_sigma = float(np.sqrt(sigma[j, :6] ** 2 @ optimizer.derivatives[j] ** 2))
            row = {"section": name, "condition": f"{state['twa']:.0f}/{state['tws']:.0f}",
                   "speed_kt": state["speed"], "success": state["success"],
                   **{f"{n}_sur": float(v) for n, v in zip(OUTPUT_NAMES, mean[j])},
                   **{f"{n}_sigma": float(v) for n, v in zip(OUTPUT_NAMES, sigma[j])},
                   "speed_error_kt": speed_error, "speed_sigma_kt": speed_sigma,
                   **{f"att{i}": float(v) for i, v in enumerate(point[optimizer.problem.n_scores:])}}
            for key, array in values.items():
                row.update({f"{n}_{key}": float(v) for n, v in zip(OUTPUT_NAMES, array[j])})
                if key not in ("hf", "lf"):
                    row[f"speed_error_{key}_kt"] = float(
                        (mean[j, :6] - array[j, :6]) @ optimizer.derivatives[j])
            rows.append(row)
    return rows
