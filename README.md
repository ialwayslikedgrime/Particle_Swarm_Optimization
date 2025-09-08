# Particle Swarm Optimization (PSO)

Implementation of **Particle Swarm Optimization (PSO)** consistent with the [`softpy`](https://github.com/andreacampagner/softpy) library, developed as part of the *Fuzzy Systems & Evolutionary Computing* course (A.Y. 2024/2025, UniMiB).  

---

## Highlights

- **Implemented from scratch**: particle dynamics, velocity updates, and swarm evolution.  
- **OOP & Design Patterns**: subclassed `FloatVectorCandidate` and applied the Factory Method.  
- **Metaheuristics**: applied swarm intelligence concepts (personal, local, global bests).  
- **Numerical Computing**: vectorized with NumPy, handled boundaries, and ensured reproducibility.  
- **Testing**: verified correctness on benchmark functions (Sphere, Rosenbrock).  

---

## Optimization Example

Visualization of PSO optimization on the **Rosenbrock function**:

![PSO Optimization Progress](pictures/test_18__pop=50_nei=10_iters=100_low=[-2.0,-2.0]_up=[2.0,2.0]_snap=5.png)

Particles (red) converge toward the global optimum (star) as iterations progress.  

---

## Quick Start

```bash
pip install -r requirements.txt

python test.py particle_candidate.py
```