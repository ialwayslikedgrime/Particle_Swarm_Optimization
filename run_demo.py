# run_demo.py
import numpy as np
from optimizer import ParticleSwarmOptimizer

# Define a simple fitness function (Sphere function)
def sphere(particle):
    return -np.sum(particle.candidate ** 2)  # maximize the negative → minimize sum of squares

if __name__ == "__main__":
    pso = ParticleSwarmOptimizer(
        fitness_func=sphere,
        pop_size=20,
        n_neighbors=3,
        size=2,
        lower=np.array([-5, -5]),
        upper=np.array([5, 5]),
    )
    pso.fit(n_iters=50)
    print("Best solution:", pso.global_best.candidate)
    print("Best fitness:", pso.global_fitness_best)