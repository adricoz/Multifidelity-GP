"""
The TRUE optimum of the digital twin (twin.py), searched directly: no surrogate to learn, the truth is
free, so a classical constrained ascent with the real VPP is affordable. This is the answer the method
(run_twin.py) should find.

Trust-region ascent from each start section z:
  1. VPP states of z on the truth (every condition);
  2. speed sensitivities d_ck = dV/dC_k at these states (VPPPool.sensitivities, finite differences),
     refreshed every `sens_every` accepted steps and before any radius cut;
  3. step: maximise the objective linearised at the frozen states of z,
         J_lin(z') = sum_c w_c sum_k d_ck m_k(z', a_c) / sum_c w_c      (m_k: the truth)
     inside the search space and the ball |z' - z| <= r (SLSQP; exact gradient at z by the envelope
     theorem);
  4. the step is CHECKED with the real VPP at z' (same conditions): accepted when the true weighted
     speed rises, the radius grows after a full accepted step and is halved after a refused one.
Stops when the radius falls below r_min, after 3 accepted steps of less than `small_gain` kt, at
max_iter or at the deadline. Each finished start is written at once to results/reference.json.

    ..\\.venv-mds\\Scripts\\python twin_experiments\\reference.py --workers 14 --deadline 07:10
    ..\\.venv-mds\\Scripts\\python twin_experiments\\reference.py --smoke --workers 4        (a few minutes)
"""
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from twin import (HERE, TRUTH_RUN, SearchDomain, TwinTruth, make_pool, mean_speed,  # noqa: E402
                  run_sections)

N_FORCES = 6


def log(message: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


def parse_deadline(text: str):
    """'HH:MM' -> the next time the clock shows it; None without a deadline."""
    if not text:
        return None
    hour, minute = (int(v) for v in text.split(":"))
    now = dt.datetime.now()
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return deadline if deadline > now else deadline + dt.timedelta(days=1)


class Reference:
    """Direct ascent on the truth (module docstring)."""

    def __init__(self, truth, domain, pool, conditions, weights, deadline=None, r_start=0.5,
                 r_max=1.0, r_min=0.03, max_iter=25, sens_every=3, small_gain=5e-4):
        self.truth, self.domain, self.pool = truth, domain, pool
        self.conditions, self.weights = conditions, np.asarray(weights, dtype=float)
        self.deadline = deadline
        self.r_start, self.r_max, self.r_min = r_start, r_max, r_min
        self.max_iter, self.sens_every, self.small_gain = max_iter, sens_every, small_gain
        self.n_solves = 0

    # --- VPP on the truth ---
    def solve(self, scores: np.ndarray, near_states=None) -> list:
        """States of one section; from the neighbour's states when given, else the robust solve
        of scripts/profile_analysis.py (fixed guess, generic guesses, second pass)."""
        if near_states is None:
            from profile_analysis import robust_solve  # noqa: PLC0415
            self.n_solves += 2 * len(self.conditions)
            return robust_solve(self.pool, [scores], [1.0], self.conditions)[0]
        self.n_solves += len(self.conditions)
        return self.pool.solve([scores], [1.0], self.conditions,
                               guesses=[s["unknowns"] for s in near_states])[0]

    def sensitivities(self, scores: np.ndarray, states: list) -> np.ndarray:
        """dV/dC (n_conditions, 6); NaN rows where the state has no equilibrium."""
        solved = [j for j, s in enumerate(states) if s["success"]]
        out = np.full((len(self.conditions), N_FORCES), np.nan)
        if solved:
            self.n_solves += 2 * N_FORCES * len(solved)
            out[solved] = self.pool.sensitivities(scores, 1.0, [self.conditions[j] for j in solved],
                                                  [states[j] for j in solved])
        return out

    # --- linearised objective at frozen states ---
    def linear_model(self, states: list, derivatives: np.ndarray):
        """J_lin(sections (n, k)) -> (n,) and its central-difference gradient at one section."""
        usable = np.array([s["success"] for s in states]) & np.all(np.isfinite(derivatives), axis=1)
        attitudes = np.array([[s["Cant_Eff"], s["Rake_Eff"], s["Yaw_Eff"]] for s in states])[usable]
        coef = self.weights[usable, None] * derivatives[usable] / self.weights[usable].sum()

        def value(sections: np.ndarray) -> np.ndarray:
            sections = np.atleast_2d(sections)
            x = np.hstack([np.repeat(sections, len(attitudes), axis=0),
                           np.tile(attitudes, (len(sections), 1))])
            means = self.truth.predict(x)[:, :N_FORCES].reshape(len(sections), len(attitudes), -1)
            return np.einsum("nck,ck->n", means, coef)

        def gradient(section: np.ndarray, step: float = 1e-3) -> np.ndarray:
            eye = step * np.eye(len(section))
            values = value(np.vstack([section + eye, section - eye]))
            return (values[:len(section)] - values[len(section):]) / (2.0 * step)

        return value, gradient

    def step(self, center: np.ndarray, value, gradient, radius: float) -> np.ndarray:
        """Maximiser of J_lin in the search space and the ball of `radius` around center, pulled
        back towards center until its decoded section is admissible."""
        result = minimize(
            lambda z: -value(z)[0], center, method="SLSQP", jac=lambda z: -gradient(z),
            constraints=[{"type": "ineq", "fun": self.domain.constraints,
                          "jac": self.domain.constraint_jacobian},
                         {"type": "ineq", "fun": lambda z: radius ** 2 - np.sum((z - center) ** 2),
                          "jac": lambda z: -2.0 * (z - center)}],
            options={"maxiter": 40, "ftol": 1e-8})
        candidate = np.array(result.x)
        for _ in range(8):
            if self.domain.admissible(candidate)[0]:
                break
            candidate = center + 0.7 * (candidate - center)
        else:
            candidate = center.copy()
        return candidate

    # --- the ascent ---
    def ascent(self, name: str, start: np.ndarray) -> dict:
        clock = time.perf_counter()
        section, radius = np.array(start, dtype=float), self.r_start
        states = self.solve(section)
        derivatives, fresh, since_sens, small, history = None, False, 0, 0, []
        stop = "max_iter"
        for iteration in range(self.max_iter):
            if self.deadline is not None and dt.datetime.now() >= self.deadline:
                stop = "deadline"
                break
            if derivatives is None or since_sens >= self.sens_every:
                derivatives, fresh, since_sens = self.sensitivities(section, states), True, 0
            value, gradient = self.linear_model(states, derivatives)
            candidate = self.step(section, value, gradient, radius)
            length = float(np.linalg.norm(candidate - section))
            record = {"iteration": iteration, "radius": radius, "step": length,
                      "fresh_sensitivities": fresh, "elapsed_min": (time.perf_counter() - clock) / 60}
            if length < 1e-3:
                history.append({**record, "accepted": False, "gain": 0.0, "predicted": 0.0})
                if not fresh:
                    derivatives = None                      # stale slopes: measure them again
                    continue
                stop = "no_step"
                break
            candidate_states = self.solve(candidate, near_states=states)
            speeds, share = mean_speed([states, candidate_states], self.weights)
            gain = float(speeds[1] - speeds[0])
            predicted = float(value(candidate)[0] - value(section)[0])
            accepted = bool(np.isfinite(gain) and gain > 1e-4)
            history.append({**record, "accepted": accepted, "gain": gain, "predicted": predicted,
                            "speed_before": float(speeds[0]), "probability_compared": share})
            log(f"  {name} it {iteration}: r {radius:.3f}, step {length:.3f}, gain {gain:+.4f} kt "
                f"(predicted {predicted:+.4f}), {'accepted' if accepted else 'refused'}, "
                f"on {100 * share:.0f} % of the probability")
            if accepted:
                section, states = candidate, candidate_states
                fresh, since_sens = False, since_sens + 1
                small = small + 1 if gain < self.small_gain else 0
                if length >= 0.9 * radius:
                    radius = min(self.r_max, 1.5 * radius)
                if small >= 3:
                    stop = "small_gains"
                    break
            elif not fresh:
                derivatives = None                          # refused with stale slopes: same radius
            else:
                radius *= 0.5
            if radius < self.r_min:
                stop = "radius"
                break
        speeds, share = mean_speed([states], self.weights)
        return {"name": name, "start": np.asarray(start).tolist(), "scores": section.tolist(),
                "distance_from_start": float(np.linalg.norm(section - start)), "stop": stop,
                "speed_own_conditions": float(speeds[0]), "probability_solved": share,
                "speeds": [s["speed"] if s["success"] else None for s in states],
                "min_margin": float(self.domain.constraints(section).min()),
                "history": history, "time_min": (time.perf_counter() - clock) / 60}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--truth-run", default=str(TRUTH_RUN))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--starts", nargs="+", default=None,
                        help="sections of the truth run (best, current_section, region_i) and 'random'; "
                             "default: best, current_section, then the other local optima, then random")
    parser.add_argument("--deadline", default="", help="HH:MM: no iteration starts after it")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--n-conditions", type=int, default=None, help="first conditions only (tests)")
    parser.add_argument("--out", default="reference")
    parser.add_argument("--smoke", action="store_true", help="1 start, 2 iterations, 6 conditions")
    args = parser.parse_args()
    if args.smoke:
        args.starts, args.max_iter = args.starts or ["best"], 2
        args.n_conditions, args.out = args.n_conditions or 6, "reference_smoke"

    truth_run = Path(args.truth_run)
    truth, domain = TwinTruth.load(truth_run), SearchDomain(truth_run)
    sections = run_sections(truth_run)
    sections["random"] = domain.sample(1, seed=11)[0]
    names = args.starts or (["best", "current_section"]
                            + [n for n in sections if n.startswith("region_")] + ["random"])
    conditions, weights = domain.conditions, domain.weights
    if args.n_conditions:
        conditions, weights = conditions[:args.n_conditions], weights[:args.n_conditions]
    pool = make_pool(args.workers, HERE / "work" / args.out)
    pool.publish(truth)
    reference = Reference(truth, domain, pool, conditions, weights, parse_deadline(args.deadline),
                          max_iter=args.max_iter)
    out = {"truth_run": truth_run.name, "truth_level": truth.level, "n_conditions": len(conditions),
           "workers": args.workers, "starts": []}
    (HERE / "results").mkdir(exist_ok=True)
    for name in names:
        if reference.deadline is not None and dt.datetime.now() >= reference.deadline:
            log(f"deadline: start {name} not run")
            break
        log(f"start {name}")
        result = reference.ascent(name, sections[name])
        log(f"start {name}: {result['stop']} after {len(result['history'])} iterations, "
            f"{result['time_min']:.0f} min, moved {result['distance_from_start']:.2f} scores, "
            f"speed {result['speed_own_conditions']:.4f} kt")
        out["starts"].append(result)
        out["n_vpp_solves"] = reference.n_solves
        (HERE / "results" / f"{args.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    pool.close()
    log("done")


if __name__ == "__main__":
    main()
