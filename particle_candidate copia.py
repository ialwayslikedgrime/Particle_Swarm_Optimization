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
                 velocity: np.ndarray,
                 inertia: float =0.729,  # added default value also such that it doesn't crush later in the generate method
                 wl: float =1/3,
                 wn: float =1/3,
                 wg: float =1/3):
        

        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        candidate_array = np.asarray(candidate, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)

        # Validate dimensions
        if len(self.lower) != size or len(self.upper) != size:
            raise ValueError("Lower and upper bounds must have same length as size")
        if len(candidate_array) != size or len(velocity_array) != size:
            raise ValueError("Candidate and velocity must have same length as size")

        # Validate bounds
        if np.any(self.lower >= self.upper):
            raise ValueError("All lower bounds must be strictly less than upper bounds")

        # Validate candidate within bounds
        if np.any(candidate_array < self.lower) or np.any(candidate_array > self.upper):
            raise ValueError("Candidate position must be within specified bounds")

        # Validate PSO weights
        wl, wn, wg = float(wl), float(wn), float(wg)
        if not np.isclose(wl + wn + wg, 1.0, rtol=1e-9):
            raise ValueError(f"PSO weights wl + wn + wg must sum to 1.0, got {wl + wn + wg}")

        # Call parent constructor with dummy distribution (PSO doesn't use it)
        super().__init__(size=size,
                         candidate=candidate_array,
                         lower=self.lower,
                         upper=self.upper,
                         distribution=stats.uniform(loc=0, scale=0),  # we know the PSO doens't use distribution specifically to ?mutate?
                         # yet we need to pass this parameter to stay consistent with the sofpy interface. 
                         # could have used a class zero distribution that basically returned 0s. I will use this, so that if someone passes extra
                         # arguments it won't crush.
                         intermediate=False
                         # mutate/recombine - I don't pass them. they are callable and I will ovverride them.
                         )

        # Store PSO-specific attributes
        self.velocity = velocity_array
        self.inertia = float(inertia)
        self.wl, self.wn, self.wg = float(wl), float(wn), float(wg)

        if not np.isclose(self.wl + self.wn + self.wg, 1.0):
            raise ValueError("wl + wn + wg must equal 1.0")

 
    @staticmethod # i guess, although there is not in the sofpy library maybe if u call class.name is not needed ?
    def generate(size: int, lower: np.ndarray, upper: np.ndarray) -> 'ParticleCandidate':
        candidate = np.random.uniform(lower, upper, size)

        # should probably insert a try/except here 
        span = np.abs(upper - lower)
        negative_span = -span

        velocity = np.random.uniform(negative_span, span, size=size)

        return ParticleCandidate(size, lower, upper, candidate, velocity)

    # i guess here we are makind the position update
    def mutate(self) -> 'ParticleCandidate':
        candidate = self.candidate + self.velocity

        # Apply boundary constraints to keep particle within search space
        # Using clipping to ensure particles don't leave the allowed region
        candidate = np.clip(candidate, self.lower, self.upper)
        return ParticleCandidate(size=self.size,
                                 lower=self.lower,
                                 upper=self.upper,
                                 candidate=candidate,
                                 velocity=self.velocity,
                                 inertia=self.inertia,
                                 wl=self.wl,
                                 wn=self.wn,
                                 wg=self.wg)  # here i should consider wether adding some sort of constratint to not let the particles go wherever in the search space

    def recombine(self, local_best, neighborhood_best, best) -> 'ParticleCandidate':
        # random vectors in [0,1]. scalar or vectors should both be acceptable
        # we are generating vectors with random numbers between [0 included and 1 excluded]
        # the number of dimensions of such vectors will be equals to size.
        # in theory, in the book 1 was included. but apparently the standard is using np.random.rand anyway. could alternatively consider:

        # rng = np.random.default_rng()
        # rl = rng.integers(0, 2**53 + 1, size=self.size) / (2**53)  # float64 grid

        # to make sure that the 1 is actually included.

        # -----------------------------
        # local_best, neighborhood_best, and best are all ParticleCandidate objects
        # Each of them has a 'candidate' attribute (which is a numpy array)

        rl = np.random.rand(self.size)
        rn = np.random.rand(self.size)
        rg = np.random.rand(self.size)

        '''formula as it is written on the assignment, but I suspect it should be the opposite? or the particle will continue to move away
         from the opimal solution ?  '''
        '''new_velocity = (self.inertia * self.velocity) \
            + rl * self.wl * (self.candidate - local_best.candidate) \
            + rn * self.wn * (self.candidate - neighborhood_best.candidate) \
            + rg * self.wg * (self.candidate - best.candidate) '''
        
        new_velocity = (self.inertia * self.velocity) \
            + rl * self.wl * (local_best.candidate - self.candidate) \
            + rn * self.wn * (neighborhood_best.candidate - self.candidate) \
            + rg * self.wg * (best.candidate - self.candidate)

        return ParticleCandidate(self.size,
                                 self.lower,
                                 self.upper,
                                 self.candidate,
                                 new_velocity)


## da qui. e ti chiedi. perrchè mutate e recombine prima del prossimo coso da implementare'?? (( dove poi in verità dovrai guardare decoratore etc.
# what does it mean with API lol.

class ParticleSwarmOptimizer(MetaHeuristicsAlgorithm):
    '''args
    pop_size: integer. it is the size of the population
    population: a list (or numpy array) of pop size ParticleCandidate
    fitness func: a fitness function that takes as input a ParticleCandidate and returns a number

    • n neighbors: an integer, the number of neighbors for each particle
    • best: an array of ParticleCandidate instances of size pop size. For
    each particle i, it must contain the position which gave the largest
    fitness for that particle
    • f itness best: a numpy array of floats, of size pop size. For each
    particle i, it must contain the largest fitness value found so far for
    that particle
    • global best: a ParticleCandidate. It must contain the position that
    gave the largest fitness value found so far
    • global f itness best: a float. It must contain the largest fitness value
    found so far

    
    '''
    def __init__(self, fitness_func, pop_size: int, n_neighbors: int, **kwargs):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.best = None
        self.fitness_best = None
        self.global_best = None
        self.global_fitness_best = None
        self.kwargs = kwargs


            
    def fit(self, n_iters):
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


            
