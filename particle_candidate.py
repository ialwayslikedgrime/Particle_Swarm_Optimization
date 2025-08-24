import numpy as np
import scipy.stats as stats
from softpy import FloatVectorCandidate


class ParticleCandidate(FloatVectorCandidate):

    # here all the parameters that we want
    def __init__(self,
                 size,
                 lower,
                 upper,
                 candidate,
                 velocity,
                 inertia = 0.729, # added default value also such that it doesn't crush later in the generate method
                 wl = 1/3,
                 wn = 1/3,
                 wg = 1/3):
        # the assignment felt a bit unclear on whether the weights had to be hardcoded as default value or had to be part of the init constructor.
        # I just gave default values and let the user free to change their values.
        
        # should re watch the inertia default value , according to literature.


         # inserting here the parameters we are inheriting from the Parent Class (they are present in floatvectorcandidate):
         # now, our particlecandidate might have some parameters (or attributes ?) that we don't want to use, given the fact that our particlecandidate 
         # might for instance not need the dimension attribute.
         # however, if we want to be consistent with the softpy interface, we need to use the method super() so that we inherit form the parent class the attributes
         # and python does not run the init method again reinstatiang them
         # while I can re initialize the methods generate etc for my own logic.
        super().__init__(size = size, 
                         candidate = candidate,
                         lower = lower, 
                         upper = upper,
                         distribution=stats.uniform(loc=0, scale=0), # we know the PSO doens't use distribution specifically to ?mutate?
                         # yet we need to pass this parameter to stay consistent with the sofpy interface. 
                         # could have used a class zero distribution that basically returned 0s. I will use this, so that if someone passes extra
                         # arguments it won't crush.
                         intermediate = False
                         # mutate/recombine - I don't pass them. they are callable and I will ovverride them.
                         )
        
        '''migliorare qui con la validation logic che avevo prima'''

        # PSO-specific attributes (THIS WILL BE WHERE I CAN INSERT THE VALIDATION LOGIC )
        self.velocity = np.asarray(velocity, dtype=float)
        self.inertia  = float(inertia)
        self.wl, self.wn, self.wg = float(wl), float(wn), float(wg)

        if not np.isclose(self.wl + self.wn + self.wg, 1.0):
            raise ValueError("wl + wn + wg must equal 1.0")
        

        # ovverriding the methods we need:
        def generate(self, size, lower, upper) -> ParticleCandidate:
            candidate = np.random.uniform(lower, upper, size)

            # should probably insert a try/except here 

            span = np.abs(upper-lower)
            negative_span = - span

            velocity = np.random.uniform(negative_span, span)

            return ParticleCandidate(size=size, lower=lower, upper=upper)



        def mutate(self) -> ParticleCandidate:
            self.candidate = self.candidate + self.velocity
            return self  ## here i should consider wether adding some sort of constratint to not let the particles go wherever in the search space
        
        
        def recombine(self, local_best, neighborhood_best, best):

            # random vectors in [0,1]. scalar or vectors should both be acceptable
            # we are generating vectors with random numbers between [0 included and 1 excluded]
            # the number of dimensions of such vectors will be equals to size.
            # in theory, in the book 1 was included. but apparently the standard is using np.random.rand anyway. could alternatively consider:

            # rng = np.random.default_rng()
            # rl = rng.integers(0, 2**53 + 1, size=self.size) / (2**53)  # float64 grid

            # to make sure that the 1 is actually included.

            rl = np.random.rand(self.size) 
            rn = np.random.rand(self.size)
            rg = np.random.rand(self.size)

            self.velocity = (self.inertia * self.velocity)
            + rl * self.wl * (self.candidate - self.local_best)
            + rn * self.wn * (self.candidate - self.neighborhood_best)
            + rg * self.wg * (self.candidate - self.best)

            return self
        

        ## da qui. e ti chiedi. perrchè mutate e recombine prima del prossimo coso da implementare'?? (( dove poi in verità dovrai guardare decoratore etc.
        # what does it mean with API lol.





        ## DA QUI
        
        # attributes that there are not in the parent class: " velocity, inertia, wl, wn, wg



        
        # Layer 3: Add PSO-specific attributes
        self.velocity = velocity
        self.inertia = inertia
        self.wl = wl
        self.wn = wn
        self.wg = wg


         # Validate PSO-specific constraints
        # Weight constraint: wl + wn + wg = 1 (fundamental PSO requirement)
        weight_sum = wl + wn + wg
        if not np.isclose(weight_sum, 1.0, rtol=1e-9):
            raise ValueError(f"PSO weight parameters wl, wn, wg must sum to 1.0, but sum to {weight_sum}")
        


        # Bounds constraint: candidate positions must be within [lower, upper]
        # This ensures particles start in valid regions of the search space
        if np.any(self.candidate < self.lower) or np.any(self.candidate > self.upper):
            raise ValueError(
                f"Particle position must be within specified bounds.\n"
                f"Lower bounds: {self.lower}\n"
                f"Upper bounds: {self.upper}\n"
                f"Current position: {self.candidate}"
            )


    @staticmethod
    def generate(size, lower, upper, inertia=0.9, wl=0.4, wn=0.3, wg=0.3):
        """
        Factory method for creating new PSO particles with random initialization.
        
        This method demonstrates the factory pattern by handling all the complex
        logic of generating appropriate random positions and velocities for PSO.
        Users simply specify the search space, and the factory delivers a
        properly configured particle ready for optimization.
        
        The static nature allows calling ParticleCandidate.generate() without
        needing an existing particle instance, which is essential for creating
        the initial swarm population.
        """
        
        # Validate inputs at the factory level as well
        validated_size = ParticleCandidate._validate_size(size)
        validated_lower = np.array(lower, dtype=float)
        validated_upper = np.array(upper, dtype=float)
        
        # Ensure lower and upper have correct dimensions
        if len(validated_lower) != validated_size:
            raise ValueError(f"Lower bounds array length ({len(validated_lower)}) must match size ({validated_size})")
        if len(validated_upper) != validated_size:
            raise ValueError(f"Upper bounds array length ({len(validated_upper)}) must match size ({validated_size})")
        
        # Ensure lower bounds are actually lower than upper bounds
        if np.any(validated_lower >= validated_upper):
            raise ValueError("All lower bounds must be strictly less than corresponding upper bounds")
        
        # Generate random candidate position within bounds
        # Using uniform distribution ensures equal probability across the search space
        candidate = np.random.uniform(validated_lower, validated_upper, size=validated_size)
        
        # Generate random velocity with magnitude proportional to search space size
        # This scaling ensures velocities are appropriate for the search space dimensions
        velocity_range = np.abs(validated_upper - validated_lower)
        velocity = np.random.uniform(-velocity_range, velocity_range, size=validated_size)
        
        # Use the constructor to create the particle, leveraging all our validation logic
        return ParticleCandidate(validated_size, validated_lower, validated_upper, 
                               candidate, velocity, inertia, wl, wn, wg)

    def mutate(self):
        """
        PSO-specific mutation: update position by adding velocity.
        
        This method overrides the parent class mutation to implement PSO's
        velocity-based movement model. Unlike generic evolutionary algorithms
        that add random noise, PSO particles move according to their velocity
        vectors, which encode both direction and magnitude of movement.
        
        Following the softpy interface pattern, this method returns a new
        ParticleCandidate rather than modifying the existing one, maintaining
        the immutable-style operations throughout the library.
        """
        
        # Calculate new position using PSO position update rule
        new_candidate = self.candidate + self.velocity
        
        # Apply boundary constraints to keep particle within search space
        # This is crucial for maintaining valid solutions during optimization
        new_candidate = np.clip(new_candidate, self.lower, self.upper)
        
        # Return new particle instance with updated position
        # Velocity remains unchanged during mutation - it's updated during recombination
        return ParticleCandidate(
            self.size, self.lower, self.upper, new_candidate, self.velocity,
            self.inertia, self.wl, self.wn, self.wg
        )

    def recombine(self, local_best, neighborhood_best, global_best):
        """
        PSO-specific recombination: update velocity based on PSO formula.
        
        This method implements the core PSO velocity update equation, which
        balances three forces:
        1. Inertia: tendency to continue in current direction
        2. Cognitive component: attraction to particle's personal best
        3. Social component: attraction to neighborhood and global best positions
        
        The method takes three different "best" positions as input, which is
        quite different from traditional recombination that typically involves
        just two parents. This reflects PSO's unique information sharing mechanism.
        """
        
        # Generate random coefficients for stochastic components
        # These ensure PSO doesn't become purely deterministic
        rl = np.random.random()  # Random coefficient for local (cognitive) component
        rn = np.random.random()  # Random coefficient for neighborhood component  
        rg = np.random.random()  # Random coefficient for global (social) component
        
        # Implement PSO velocity update formula
        # This is the heart of the PSO algorithm, encoding how particles learn
        # from their own experience and social information
        new_velocity = (
            self.inertia * self.velocity +                                    # Inertia component
            rl * self.wl * (local_best.candidate - self.candidate) +         # Cognitive component
            rn * self.wn * (neighborhood_best.candidate - self.candidate) +  # Neighborhood component
            rg * self.wg * (global_best.candidate - self.candidate)          # Social component
        )
        
        # Return new particle instance with updated velocity
        # Position remains unchanged during recombination - it's updated during mutation
        return ParticleCandidate(
            self.size, self.lower, self.upper, self.candidate, new_velocity,
            self.inertia, self.wl, self.wn, self.wg
        )

    def __str__(self):
        """ additional """
     
        return (f"ParticleCandidate(position={self.candidate}, "
                f"velocity={self.velocity}, fitness=?)")

    def __repr__(self):
        """ additional """
       
        return (f"ParticleCandidate(size={self.size}, "
                f"position={self.candidate}, velocity={self.velocity}, "
                f"inertia={self.inertia}, weights=({self.wl}, {self.wn}, {self.wg}))")


        
