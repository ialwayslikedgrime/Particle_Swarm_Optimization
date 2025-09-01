import numpy as np
from particle_candidate import ParticleCandidate, ParticleSwarmOptimizer  # Replace with your actual filename

# Test 1: Simple 2D optimization problem - maximize -(x^2 + y^2)
# The optimal solution should be at (0, 0) with fitness value 0
def simple_fitness(particle):
    """Maximize -(x^2 + y^2) - optimal at (0,0)"""
    x, y = particle.candidate
    return -(x**2 + y**2)

print("Test 1: Simple 2D quadratic function")
print("Expected optimum: (0, 0) with fitness ≈ 0")

optimizer = ParticleSwarmOptimizer(
    fitness_func=simple_fitness,
    pop_size=20,
    n_neighbors=5,
    size=2,
    lower=np.array([-5.0, -5.0]),
    upper=np.array([5.0, 5.0])
)

optimizer.fit(50)

print(f"Best position found: {optimizer.global_best.candidate}")
print(f"Best fitness: {optimizer.global_fitness_best}")
print(f"Distance from optimum: {np.linalg.norm(optimizer.global_best.candidate)}")
print()

# Test 2: Sphere function in 5D
def sphere_fitness(particle):
    """Maximize -sum(x_i^2) - optimal at origin"""
    return -np.sum(particle.candidate**2)

print("Test 2: 5D Sphere function")
print("Expected optimum: (0, 0, 0, 0, 0) with fitness ≈ 0")

optimizer2 = ParticleSwarmOptimizer(
    fitness_func=sphere_fitness,
    pop_size=30,
    n_neighbors=7,
    size=5,
    lower=np.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
    upper=np.array([3.0, 3.0, 3.0, 3.0, 3.0])
)

optimizer2.fit(100)

print(f"Best position found: {optimizer2.global_best.candidate}")
print(f"Best fitness: {optimizer2.global_fitness_best}")
print(f"Distance from optimum: {np.linalg.norm(optimizer2.global_best.candidate)}")
print()

# Test 3: Check if algorithm improves over iterations
def track_progress_fitness(particle):
    return -(particle.candidate[0]**2 + particle.candidate[1]**2)

print("Test 3: Progress tracking")
optimizer3 = ParticleSwarmOptimizer(
    fitness_func=track_progress_fitness,
    pop_size=15,
    n_neighbors=3,
    size=2,
    lower=np.array([-2.0, -2.0]),
    upper=np.array([2.0, 2.0])
)

# Track fitness over iterations
fitness_history = []
for i in range(30):
    if i == 0:
        optimizer3.fit(1)
    else:
        # Continue optimization
        for iteration in range(1):
            current_fitness = []
            for j, particle in enumerate(optimizer3.population):
                fitness = optimizer3.fitness_func(particle)
                current_fitness.append(fitness)
                
                if fitness > optimizer3.fitness_best[j]:
                    optimizer3.fitness_best[j] = fitness
                    optimizer3.best[j] = particle
                    
                if fitness > optimizer3.global_fitness_best:
                    optimizer3.global_fitness_best = fitness
                    optimizer3.global_best = particle
            
            best_neighbors = []
            for j in range(optimizer3.pop_size):
                possible_neighbors = list(range(optimizer3.pop_size))
                possible_neighbors.remove(j)
                neighbor_indices = np.random.choice(possible_neighbors, 
                                                  min(optimizer3.n_neighbors, len(possible_neighbors)), 
                                                  replace=False)
                
                best_neighbor_idx = neighbor_indices[np.argmax([optimizer3.fitness_best[k] for k in neighbor_indices])]
                best_neighbors.append(optimizer3.best[best_neighbor_idx])
            
            for j in range(optimizer3.pop_size):
                updated_particle = optimizer3.population[j].recombine(optimizer3.best[j], best_neighbors[j], optimizer3.global_best)
                optimizer3.population[j] = updated_particle.mutate()
    
    fitness_history.append(optimizer3.global_fitness_best)
    if i % 10 == 0:
        print(f"Iteration {i}: Best fitness = {optimizer3.global_fitness_best:.6f}")

print(f"Final best fitness: {optimizer3.global_fitness_best:.6f}")
print(f"Fitness improved: {fitness_history[-1] > fitness_history[0]}")

print("\n=== All tests completed ===")