import numpy as np
import scipy.stats as stats
from softpy import FloatVectorCandidate


class ParticleCandidate(FloatVectorCandidate):
    """
    PSO Particle implementation that inherits from FloatVectorCandidate.

    This class represents a particle in Particle Swarm Optimization, extending
    the base FloatVectorCandidate with PSO-specific attributes (velocity, inertia, weights)
    and implementing PSO-specific mutation and recombination logic.
    """

    def __init__(self,
                 size,
                 lower,
                 upper,
                 candidate,
                 velocity,
                 inertia=0.729,  # added default value also such that it doesn't crush later in the generate method
                 wl=1/3,
                 wn=1/3,
                 wg=1/3):
        """
        Initialize a PSO particle.

        Args:
            size: Number of dimensions in the search space
            lower: Lower bounds for each dimension (numpy array)
            upper: Upper bounds for each dimension (numpy array) 
            candidate: Current position of the particle (numpy array)
            velocity: Current velocity of the particle (numpy array)
            inertia: Inertia weight for velocity update
            wl: Weight for cognitive (local best) component
            wn: Weight for neighborhood best component  
            wg: Weight for global best component
        """
        # the assignment felt a bit unclear on whether the weights had to be hardcoded as default value or had to be part of the init constructor.
        # I just gave default values and let the user free to change their values.
        # should re watch the inertia default value , according to literature.

        # inserting here the parameters we are inheriting fr
        # om the Parent Class (they are present in floatvectorcandidate):
        # now, our particlecandidate might have some parameters (or attributes ?) that we don't want to use, given the fact that our particlecandidate 
        # might for instance not need the dimension attribute.
        # however, if we want to be consistent with the softpy interface, we need to use the method super() so that we inherit form the parent class the attributes
        # and python does not run the init method again reinstatiang them
        # while I can re initialize the methods generate etc for my own logic.

        # Convert inputs to proper numpy arrays
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

        new_velocity = (self.inertia * self.velocity) \
            + rl * self.wl * (self.candidate - local_best.candidate) \
            + rn * self.wn * (self.candidate - neighborhood_best.candidate) \
            + rg * self.wg * (self.candidate - best.candidate)

        return ParticleCandidate(self.size,
                                 self.lower,
                                 self.upper,
                                 self.candidate,
                                 new_velocity)


## da qui. e ti chiedi. perrchè mutate e recombine prima del prossimo coso da implementare'?? (( dove poi in verità dovrai guardare decoratore etc.
# what does it mean with API lol.

