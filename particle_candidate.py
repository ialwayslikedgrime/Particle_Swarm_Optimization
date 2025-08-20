import numpy as np
from softpy import FloatVectorCandidate


class ParticleCandidate(FloatVectorCandidate):
            
    def _validate_array_parameter(self, parameter, size, param_name):
        """function to validate array parameters with consistent logic."""
        
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



    # valori default da rivedersi. permette di non doverli imputare ogni singola volta ma non scritti da nessuna parte.
    def __init__(self, size: int, lower, upper, candidate, velocity, inertia=0.9, wl=0.4, wn=0.3, wg=0.3):
        # Size validation
        if not isinstance(size, int):
            raise TypeError("size should be an integer")
        elif size < 1:
            raise ValueError("the number of components should be at least 1")

        super().__init__()
        self.size = size


        # Use helper function for all array validations
        self.lower = self._validate_array_parameter(lower, size, "lower")
        self.upper = self._validate_array_parameter(upper, size, "upper") 
        self.candidate = self._validate_array_parameter(candidate, size, "candidate")
        self.velocity = self._validate_array_parameter(velocity, size, "velocity")









        self.inertia = inertia # dtype = float. can I insert dype ?
        self.wl = wl
        self.wn = wn
        self.wg = wg

        #validazione del constraint: wl +wn +wg = 1
        somma_pesi = wl + wn + wg
        if not np.isclose(somma_pesi, 1.0, rtol=1e-9):
            raise ValueError(f"I pesi wl, wn, wg devono sommare a 1.0, ma sommano a {somma_pesi}")
        
        
        # Validazione che candidate sia entro i limiti
        if np.any(self.candidate < self.lower) or np.any(self.candidate > self.upper):
            raise ValueError(
                f"La posizione candidate deve essere entro i limiti specificati.\n"
                f"Lower: {self.lower}\n"
                f"Upper: {self.upper}\n"
                f"Candidate: {self.candidate}"
            )

        
        # a specification is that our candidate should be represented by a vector that has a number of dimension equal to the size.
        # ricordati di quel punto. ma forse lo posso definire dopo o si definisce in automatico?

    
        # dobbiamo stare attenti agli errori di precisione in virgola mobile. Quando Python calcola operazioni con numeri decimali, 
        # a volte il risultato non è esattamente quello che ci aspettiamo. Per esempio, 0.1 + 0.2 in Python non è esattamente 0.3, 
        # ma qualcosa come 0.30000000000000004.

        # La funzione np.isclose è molto più robusta di un semplice confronto con == per i numeri in virgola mobile. 
        # Questa funzione accetta un parametro rtol (relative tolerance) che specifica quanto possono differire 
        # due numeri pur essendo considerati "vicini abbastanza" da essere uguali.


    def generate(self, size, lower, upper):

        candidate = np.array()
        velocity = np.array()

        """generate: it must accept size, lower, upper as input parameters. It
        must generate a candidate (numpy array of size cells) by drawing
        values uniformly at random between lower and upper. It then must
        generate a velocity (numpy array of size cells) by drawing values
        uniformly at random between −|upper−lower|and |upper−lower|.
        It must then call the constructor with the given input parameters
        and return its result."""


        return candidate

    def mutate(self):
        candidate = candidate + self.velocity
    
    def recombine(self, local_best, neighborhood_best, best):
        velocity = inertia * velocity \
            + rl * wl * (candidate - local_best) \
            + rn * wn * (candidate - neighborhood_best) \
            + rg * wg * (candidate − best)
        
        return velocity
 

    
   
