import numpy as np
import scipy.stats as stats
from softpy import FloatVectorCandidate


class ParticleCandidate(FloatVectorCandidate):
    """
    PSO particle implementation that inherits from FloatVectorCandidate.
    """
            
    def _validate_array_parameter(self, parameter, size, param_name):
        """function to validate array parameters with consistent logic.
        
         This validation serves as the first line of defense against malformed
        inputs, providing clear error messages that help users understand
        PSO requirements and constraints.
        """
        
        # Check if it's a list or numpy array
        if not isinstance(parameter, (list, np.ndarray)):
            raise TypeError(f"{param_name} should be a list or numpy array")
        
        # Check if it has the correct size
        elif len(parameter) != size:
            raise ValueError(f"{param_name} should have exactly {size} elements but it actually has {len(parameter)}")
        
        # Try to convert to numpy array with float dtype
        try:
            validated_array = np.array(parameter, dtype=float)
            return validated_array
        except (ValueError, TypeError) as e:
            raise ValueError(f"{param_name} contains invalid values that cannot be converted to float")
    

    @staticmethod
    def _validate_size(size):
        """
        Static method to validate the size parameter.
        """
         
        if not isinstance(size, int):
            raise TypeError("size should be an integer")
        elif size < 1:
            raise ValueError("the number of components should be at least 1")
        return size



   # valori default da rivedersi. permette di non doverli imputare ogni singola volta ma non scritti da nessuna parte.
    def __init__(self, size: int, lower, upper, candidate, velocity, inertia=0.9, wl=0.4, wn=0.3, wg=0.3):
        """
        Constructor for PSO particles.

        This constructor demonstrates the layered approach to object creation:
        1. First, validate all inputs to catch problems early
        2. Then, call the parent constructor to set up basic structure
        3. Finally, add PSO-specific attributes and constraints
        """

        # Layer 1: Input validation - catch problems before they propagate
        validated_size = self._validate_size(size)
        validated_lower = self._validate_array_parameter(lower, size, "lower")
        validated_upper = self._validate_array_parameter(upper, size, "upper")
        validated_candidate = self._validate_array_parameter(candidate, size, "candidate")
        validated_velocity = self._validate_array_parameter(velocity, size, "velocity")


        # Layer 2: Call parent constructor to establish inheritance relationship

        super().__init__(size = validated_size, 
                         candidate = validated_candidate,
                         distribution=stats.norm(0, 1),  # Dummy distribution - PSO won't use this
                         lower = validated_lower,
                         upper = validated_upper,
                         intermediate=False,  # PSO doesn't use intermediate recombination
                         mutate=None,        # We'll override with PSO-specific mutate
                         recombine=None      # We'll override with PSO-specific recombine  
                )
        
        # Layer 3: Add PSO-specific attributes
        self.velocity = validated_velocity
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
        """
        String representation for debugging and visualization.
        
        This method isn't required by the assignment but demonstrates good
        software engineering practice by making objects easier to inspect
        and debug during development.
        """
        return (f"ParticleCandidate(position={self.candidate}, "
                f"velocity={self.velocity}, fitness=?)")

    def __repr__(self):
        """
        Detailed representation for development and debugging.
        """
        return (f"ParticleCandidate(size={self.size}, "
                f"position={self.candidate}, velocity={self.velocity}, "
                f"inertia={self.inertia}, weights=({self.wl}, {self.wn}, {self.wg}))")


        
