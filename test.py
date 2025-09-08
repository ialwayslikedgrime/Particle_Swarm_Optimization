"""
PSO Test Harness — uses the user's own ParticleCandidate & ParticleSwarmOptimizer.

Usage:
  python pso_test.py my_impl.py
  python pso_test.py my_impl
"""

import sys
import time
import importlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- dynamic import ----------------------
def load_user_impl(arg):
    if arg.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_pso_impl", arg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    else:
        module = importlib.import_module(arg)

    try:
        PC = getattr(module, "ParticleCandidate")
        PSO = getattr(module, "ParticleSwarmOptimizer")
    except AttributeError as e:
        raise RuntimeError(
            "Your module must export ParticleCandidate and ParticleSwarmOptimizer."
        ) from e
    return PC, PSO, module


# ---------------------- helpers ----------------------
def maybe(obj, method_name, *args, **kwargs):
    """
    Call method if it exists and return the updated object, handling
    both in-place and return-new styles.
    """
    meth = getattr(obj, method_name)
    out = meth(*args, **kwargs)
    return obj if out is None else out


# ---------------------- tests ----------------------
def test_particle_initialization(ParticleCandidate):
    print("\n[TEST 1] ParticleCandidate initialization\n" + "-" * 40)
    try:
        size = 2
        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])
        candidate = np.array([0.0, 0.0])
        velocity = np.array([1.0, -1.0])

        p = ParticleCandidate(size, lower, upper, candidate, velocity)

        assert p.size == size, "Size mismatch"
        assert p.candidate.shape == (size,), "Candidate shape mismatch"
        assert p.velocity.shape == (size,), "Velocity shape mismatch"
        assert np.all(p.candidate >= lower) and np.all(p.candidate <= upper), \
            "Candidate out of bounds"

        if all(hasattr(p, a) for a in ("wl", "wn", "wg")):
            print(f"  weights: wl={p.wl:.3f}, wn={p.wn:.3f}, wg={p.wg:.3f}")
        if hasattr(p, "inertia"):
            print(f"  inertia: {p.inertia}")

        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_generate(ParticleCandidate):
    print("\n[TEST 2] ParticleCandidate.generate()\n" + "-" * 40)
    try:
        size = 3
        lower = np.array([-10.0, -10.0, -10.0])
        upper = np.array([10.0, 10.0, 10.0])

        p = ParticleCandidate.generate(size, lower, upper)

        assert p.size == size
        assert np.all(p.candidate >= lower) and np.all(p.candidate <= upper)
        assert p.velocity.shape == (size,)
        print("  position:", p.candidate)
        print("  velocity:", p.velocity)
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_boundary_handling(ParticleCandidate):
    print("\n[TEST 3] Boundary handling in mutate()\n" + "-" * 40)
    try:
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        candidate = np.array([0.9, 0.9])
        velocity = np.array([5.0, 5.0])

        p = ParticleCandidate(2, lower, upper, candidate, velocity)
        p = maybe(p, "mutate")

        assert np.all(p.candidate >= lower) and np.all(p.candidate <= upper), \
            "Mutate left candidate out of bounds"
        print("  new position:", p.candidate)
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_optimizer_on_sphere(ParticleCandidate, ParticleSwarmOptimizer):
    print("\n[TEST 4] Optimizer on Sphere (maximization)\n" + "-" * 40)
    try:
        # Sphere (minimum at 0); return negative to turn into maximization.
        def sphere_fit(p):
            return -np.sum(p.candidate ** 2)

        opt = ParticleSwarmOptimizer(
            fitness_func=sphere_fit,
            pop_size=20,
            n_neighbors=5,
            size=2,
            lower=np.array([-5.0, -5.0]),
            upper=np.array([5.0, 5.0]),
        )
        t0 = time.time()
        opt.fit(n_iters=50)
        dt = time.time() - t0

        best = opt.global_best.candidate
        dist = float(np.linalg.norm(best - np.array([0.0, 0.0])))
        print(f"  best position: {best}")
        print(f"  distance to [0,0]: {dist:.6f}")
        print(f"  time: {dt:.3f}s")
        print("✓ PASS (ran to completion)")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_formula_direction(ParticleCandidate):
    print("\n[TEST 5] Velocity direction check\n" + "-" * 40)
    try:
        size = 1
        lower = np.array([-10.0])
        upper = np.array([10.0])

        x = np.array([5.0])
        v = np.array([0.0])

        p = ParticleCandidate(size, lower, upper, x, v)

        zero = np.array([0.0])
        lb = ParticleCandidate(size, lower, upper, zero, v.copy())
        nb = ParticleCandidate(size, lower, upper, zero, v.copy())
        gb = ParticleCandidate(size, lower, upper, zero, v.copy())

        np.random.seed(42)
        p = maybe(p, "recombine", lb, nb, gb)

        print(f"  x=5.0, best=0.0, new v={p.velocity[0]:.4f}")
        if p.velocity[0] < 0:
            print("  → STANDARD PSO (toward best)")
        else:
            print("  → ASSIGNMENT formula (away from best)")
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


# ---------------------- visualization (no PSO logic added) ----------------------
class FitnessHistory:
    """
    Wrap a fitness function to record per-iteration positions without touching
    the optimizer's logic. We infer iteration boundaries by counting calls,
    assuming the optimizer evaluates exactly pop_size particles per iteration.
    """
    def __init__(self, base_func, pop_size, snapshot_every=5):
        self.base_func = base_func
        self.pop_size = int(pop_size)
        self.snapshot_every = int(snapshot_every)
        self.call_count = 0
        self.iteration = 0
        self.pos_buf = []
        self.fit_buf = []
        self.history = []
        self.global_best_fit = -np.inf
        self.global_best_pos = None

    def __call__(self, particle):
        f = self.base_func(particle)
        self.pos_buf.append(particle.candidate.copy())
        self.fit_buf.append(f)
        self.call_count += 1

        # end of an iteration:
        if self.call_count % self.pop_size == 0:
            # update running global best using this iteration's evaluations
            k = int(np.argmax(self.fit_buf))
            if self.fit_buf[k] > self.global_best_fit:
                self.global_best_fit = self.fit_buf[k]
                self.global_best_pos = self.pos_buf[k].copy()

            if self.iteration % self.snapshot_every == 0:
                self.history.append({
                    "iteration": self.iteration,
                    "positions": np.array(self.pos_buf),
                    "global_best": None if self.global_best_pos is None else self.global_best_pos.copy(),
                })
            # reset buffers for next iteration
            self.pos_buf = []
            self.fit_buf = []
            self.iteration += 1

        return f


def visualize_rosenbrock(ParticleSwarmOptimizer):
    print("\n[VISUALIZATION] Rosenbrock 2D (no PSO override)\n" + "-" * 40)

    # Rosenbrock; return negative to maximize
    def rosen(p):
        x, y = p.candidate[0], p.candidate[1]
        return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    pop_size = 20
    recorder = FitnessHistory(rosen, pop_size=pop_size, snapshot_every=5)

    opt = ParticleSwarmOptimizer(
        fitness_func=recorder,   # <-- ONLY wrapping, not changing PSO
        pop_size=pop_size,
        n_neighbors=5,
        size=2,
        lower=np.array([-2.0, -2.0]),
        upper=np.array([2.0, 2.0]),
    )
    opt.fit(n_iters=100)

    # Plot snapshots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("PSO Optimization Progress on Rosenbrock Function", fontsize=16)
    snaps = [0, 2, 4, 6, 8, -1]

    X = np.linspace(-2, 2, 200)
    Y = np.linspace(-2, 2, 200)
    XX, YY = np.meshgrid(X, Y)
    ZZ = (1 - XX) ** 2 + 100 * (YY - XX ** 2) ** 2

    for idx, h in enumerate(snaps):
        ax = axes[idx // 3, idx % 3]
        if h < len(recorder.history):
            data = recorder.history[h]
            P = data["positions"]
            ax.contour(XX, YY, ZZ, levels=np.logspace(-1, 3, 20), alpha=0.3)
            ax.scatter(P[:, 0], P[:, 1], c="red", s=30, alpha=0.7)
            if data["global_best"] is not None:
                ax.scatter(data["global_best"][0], data["global_best"][1],
                           c="yellow", s=100, marker="*", edgecolors="black", linewidth=2)
            ax.scatter(1, 1, c="green", s=100, marker="D", edgecolors="black", linewidth=2)
            ax.set_title(f"Iteration {data['iteration']}")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
            ax.set_xlabel("X"); ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("pso_optimization_progress.png", dpi=120, bbox_inches="tight")
    print("✓ Saved: pso_optimization_progress.png")
    print(f"Final best (from optimizer): {opt.global_best.candidate}")
    print(f"Error vs [1,1]: {np.linalg.norm(opt.global_best.candidate - np.array([1.0, 1.0])):.4f}")
    plt.show()
    return True


# ---------------------- main ----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pso_test.py <module_name | path/to/file.py>")
        sys.exit(1)

    ParticleCandidate, ParticleSwarmOptimizer, module = load_user_impl(sys.argv[1])
    print("=" * 60)
    print(f"Testing PSO implementation from: {module.__name__}")
    print("=" * 60)

    np.random.seed(0)  # reproducible tests (doesn't alter your logic)

    tests = [
        lambda: test_particle_initialization(ParticleCandidate),
        lambda: test_generate(ParticleCandidate),
        lambda: test_boundary_handling(ParticleCandidate),
        lambda: test_optimizer_on_sphere(ParticleCandidate, ParticleSwarmOptimizer),
        lambda: test_formula_direction(ParticleCandidate),
    ]

    passed = failed = 0
    for t in tests:
        ok = t()
        passed += int(ok)
        failed += int(not ok)

    visualize_rosenbrock(ParticleSwarmOptimizer)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 60)


if __name__ == "__main__":
    main()