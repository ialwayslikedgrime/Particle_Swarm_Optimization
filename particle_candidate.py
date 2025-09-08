import numpy as np
import scipy.stats as stats
from softpy import FloatVectorCandidate
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm


class ParticleCandidate(FloatVectorCandidate):
    '''
    Implementation of PSO particles inheriting from FloatVectorCandidate.
    
    This class represents a particle in Particle Swarm Optimization, extending
    the base FloatVectorCandidate with PSO-specific attributes (velocity, inertia, weights)
    and implementing PSO-specific mutation and recombination logic.
    
    Parameters
    ----------
    :param size: the number of components in the position
    :type size: int
    
    :param lower: lower bounds for each position component
    :type lower: np.ndarray
    
    :param upper: upper bounds for each position component  
    :type upper: np.ndarray
    
    :param candidate: the current position of the particle
    :type candidate: np.ndarray
    
    :param velocity: the current velocity vector of the particle
    :type velocity: np.ndarray
    
    :param inertia: inertia weight for velocity updates
    :type inertia: float, default=0.729
    
    :param wl: cognitive component weight (personal best influence)
    :type wl: float, default=1/3
    
    :param wn: local social component weight (neighborhood best influence)
    :type wn: float, default=1/3
    
    :param wg: global social component weight (global best influence)
    :type wg: float, default=1/3
    '''

    def __init__(self,
                 size: int,
                 lower: np.ndarray, 
                 upper: np.ndarray,
                 candidate: np.ndarray,
                 velocity: np.ndarray):
        

        lower= np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        candidate = np.asarray(candidate, dtype=float)
        velocity = np.asarray(velocity, dtype=float)

        # Validate dimensions
        if len(lower) != size or len(upper) != size:
            raise ValueError("Lower and upper bounds must have same length as size")
        if len(candidate) != size or len(velocity) != size:
            raise ValueError("Candidate and velocity must have same length as size")
        
        # Validate bounds
        if np.any(lower >= upper):
            raise ValueError("All lower bounds must be strictly less than upper bounds")
        
        # Validate candidate within bounds
        if np.any(candidate < lower) or np.any(candidate > upper):
            raise ValueError("Candidate position must be within specified bounds")

        # Initialize parent class (distribution required by softpy interface but unused in PSO)
        super().__init__(size=size,
                         candidate=candidate,
                         lower=lower,
                         upper=upper,
                         distribution=stats.uniform(loc=0, scale=0),
                         intermediate=False)


        # Store velocity
        self.velocity = velocity

        # Initialize PSO-specific attributes with fixed values
        self.inertia = 0.729 # Standard PSO inertia weight
        self.wl =1/3
        self.wn =1/3
        self.wg =1/3
 

    @staticmethod
    def generate(size: int, lower: np.ndarray, upper: np.ndarray) -> 'ParticleCandidate':
        candidate = np.random.uniform(lower, upper, size)

        span = np.abs(upper - lower)
        negative_span = -span

        velocity = np.random.uniform(negative_span, span, size=size)

        return ParticleCandidate(size, lower, upper, candidate, velocity)



    def mutate(self) -> 'ParticleCandidate':
        self.candidate = self.candidate + self.velocity

        # (not in the assignment)
        # Apply boundary constraints to keep particle within search space
        # # Using clipping approach: particles that would move outside bounds
        # are repositioned to the nearest boundary (conservative approach)
        self.candidate = np.clip(self.candidate, self.lower, self.upper)

        return self
        


    def recombine(self, local_best, neighborhood_best, best) -> 'ParticleCandidate':
        # Generate random coefficients for PSO velocity update
        rl = np.random.rand(self.size)
        rn = np.random.rand(self.size)
        rg = np.random.rand(self.size)

        '''formula as it is written on the assignment, but I suspect it should be the opposite? or the particle will continue to move away
         from the opimal solution ?  '''
        '''new_velocity = (self.inertia * self.velocity) \
            + rl * self.wl * (self.candidate - local_best.candidate) \
            + rn * self.wn * (self.candidate - neighborhood_best.candidate) \
            + rg * self.wg * (self.candidate - best.candidate) '''
        
        self.velocity = (self.inertia * self.velocity) \
            + rl * self.wl * (local_best.candidate - self.candidate) \
            + rn * self.wn * (neighborhood_best.candidate - self.candidate) \
            + rg * self.wg * (best.candidate - self.candidate)
        
        return self



class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    '''
    Implementation of Particle Swarm Optimization inheriting from MetaHeuristicsAlgorithm.
    
    This class implements the PSO meta-heuristic algorithm maintaining a population
    of particles that explore the search space based on personal, neighborhood, and global best positions.
    
    Parameters
    ----------
    :param fitness_func: fitness function that takes a ParticleCandidate and returns a number
    :type fitness_func: Callable
    
    :param pop_size: the size of the population
    :type pop_size: int
    
    :param n_neighbors: the number of neighbors for each particle
    :type n_neighbors: int
    
    :param kwargs: additional parameters passed to ParticleCandidate.generate()
    :type kwargs: dict
    
    Attributes
    ----------
    pop_size : int
        The size of the population
    population : list
        List of ParticleCandidate instances of size pop_size
    fitness_func : Callable
        Fitness function that evaluates ParticleCandidate instances
    n_neighbors : int
        Number of neighbors for each particle
    best : list
        Array of ParticleCandidate instances storing personal best positions
    fitness_best : np.ndarray
        Array of floats storing personal best fitness values
    global_best : ParticleCandidate
        Position with the largest fitness value found so far
    global_fitness_best : float
        The largest fitness value found so far    
    '''

    def __init__(self, fitness_func, pop_size: int, n_neighbors: int, **kwargs):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.best = None
        self.fitness_best = None
        self.population = None
        self.global_best = None
        self.global_fitness_best = None
        self.kwargs = kwargs


    def fit(self, n_iters: int) -> None:
        """
        Run the PSO optimization algorithm for a specified number of iterations.
        
        Parameters
        ----------
        n_iters : int
            Number of iterations to run the optimization
        """
        # Initialize population
        self.population = []
        for i in range(self.pop_size):
            new_particle = ParticleCandidate.generate(**self.kwargs)
            self.population.append(new_particle)
        
        # Initialize tracking arrays
        self.best = self.population.copy()
        self.fitness_best = np.full(self.pop_size, -np.inf)
        self.global_fitness_best = -np.inf
        
        # Main optimization loop
        for iteration in range(n_iters):
            # (a) Compute fitness for each particle
            current_fitness = []
            for i, particle in enumerate(self.population):
                fitness = self.fitness_func(particle)
                current_fitness.append(fitness)
                
                # (b) Update personal and global bests
                if fitness > self.fitness_best[i]:
                    self.fitness_best[i] = fitness
                    self.best[i] = particle
                    
                if fitness > self.global_fitness_best:
                    self.global_fitness_best = fitness
                    self.global_best = particle
            
            # (c) Find neighborhood bests
            best_neighbors = []
            for i in range(self.pop_size):
                possible_neighbors = list(range(self.pop_size))
                possible_neighbors.remove(i)
                neighbor_indices = np.random.choice(possible_neighbors, 
                                                min(self.n_neighbors, len(possible_neighbors)), 
                                                replace=False)
                
                best_neighbor_idx = neighbor_indices[np.argmax([self.fitness_best[j] for j in neighbor_indices])]
                best_neighbors.append(self.best[best_neighbor_idx])
            
            # (d) Update velocities and positions
            for i in range(self.pop_size):
                updated_particle = self.population[i].recombine(self.best[i], best_neighbors[i], self.global_best)
                self.population[i] = updated_particle.mutate()


            
