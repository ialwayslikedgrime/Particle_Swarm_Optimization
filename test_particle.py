import numpy as np
import sys

# Modify this import line based on where you saved your class
try:
    from particle_candidate import ParticleCandidate  # Replace with your actual filename
except ImportError:
    print("ERROR: Could not import ParticleCandidate. Make sure it's in the same directory.")
    print("TIP: If your class is in 'pso.py', change import to: from pso import ParticleCandidate")
    sys.exit(1)

def test_particle_creation():
    """Test basic particle creation using factory method"""
    
    print("Test 1: Testing particle creation...")
    
    particle = ParticleCandidate.generate(
        size=3,
        lower=[0, 0, 0], 
        upper=[10, 10, 10]
    )
    print(f"   Created particle: position={particle.candidate}")
    print(f"   Velocity: {particle.velocity}")
    print(f"   Weights sum to: {particle.wl + particle.wn + particle.wg}")
    
    # Verify basic properties
    assert particle.size == 3
    assert len(particle.candidate) == 3
    assert len(particle.velocity) == 3
    assert np.all(particle.candidate >= 0) and np.all(particle.candidate <= 10)
    assert abs((particle.wl + particle.wn + particle.wg) - 1.0) < 1e-9
    
    print("   PASSED: Basic particle creation")
    return particle

def test_particle_movement(particle):
    """Test particle movement (mutation)"""
    
    print("\nTest 2: Testing particle movement...")
    
    old_position = particle.candidate.copy()
    moved_particle = particle.mutate()
    
    print(f"   Old position: {old_position}")
    print(f"   New position: {moved_particle.candidate}")
    print(f"   Movement vector: {moved_particle.candidate - old_position}")
    
    # Verify movement properties
    assert not np.allclose(old_position, moved_particle.candidate, rtol=1e-10)
    assert np.allclose(moved_particle.velocity, particle.velocity)  # Velocity unchanged
    
    print("   PASSED: Particle movement")
    return moved_particle

def test_velocity_update(particle):
    """Test velocity update (recombination)"""
    
    print("\nTest 3: Testing velocity update...")
    
    # Create "best" positions for the particle to learn from
    local_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
    neighborhood_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
    global_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
    
    old_velocity = particle.velocity.copy()
    updated_particle = particle.recombine(local_best, neighborhood_best, global_best)
    
    print(f"   Old velocity: {old_velocity}")
    print(f"   New velocity: {updated_particle.velocity}")
    print(f"   Position unchanged: {np.allclose(particle.candidate, updated_particle.candidate)}")
    
    # Verify velocity update properties
    assert np.allclose(particle.candidate, updated_particle.candidate)  # Position unchanged
    assert not np.allclose(old_velocity, updated_particle.velocity, rtol=1e-10)  # Velocity changed
    
    print("   PASSED: Velocity update")
    return updated_particle

def test_boundary_handling():
    """Test boundary constraint handling"""
    
    print("\nTest 4: Testing boundary handling...")
    
    # Create particle near boundary with large velocity
    boundary_particle = ParticleCandidate(
        size=2,
        lower=[0, 0],
        upper=[5, 5], 
        candidate=[4.8, 4.9],  # Near upper bounds
        velocity=[2, 2]        # Would go outside bounds
    )
    
    bounded_particle = boundary_particle.mutate()
    expected_unbounded = boundary_particle.candidate + boundary_particle.velocity
    
    print(f"   Before bounds: position would be {expected_unbounded}")
    print(f"   After clipping: {bounded_particle.candidate}")
    print(f"   Stays within bounds: {np.all(bounded_particle.candidate <= [5,5])}")
    
    # Verify boundary enforcement
    assert np.all(bounded_particle.candidate <= [5, 5])
    assert np.all(bounded_particle.candidate >= [0, 0])
    
    print("   PASSED: Boundary handling")

def test_comprehensive_validation():
    """Test all validation logic thoroughly"""
    
    print("\nTest 5: Testing comprehensive input validation...")
    
    validation_tests = [
        # Size validation tests
        {
            'name': 'Invalid size type (string)',
            'test': lambda: ParticleCandidate.generate("not_int", [0], [1]),
            'expected_error': TypeError
        },
        {
            'name': 'Invalid size type (float)', 
            'test': lambda: ParticleCandidate.generate(2.5, [0], [1]),
            'expected_error': TypeError
        },
        {
            'name': 'Negative size',
            'test': lambda: ParticleCandidate.generate(-1, [0], [1]),
            'expected_error': ValueError
        },
        {
            'name': 'Zero size',
            'test': lambda: ParticleCandidate.generate(0, [0], [1]),
            'expected_error': ValueError
        },
        
        # Array dimension validation tests
        {
            'name': 'Mismatched lower bounds length',
            'test': lambda: ParticleCandidate.generate(3, [0, 0], [1, 1, 1]),
            'expected_error': ValueError
        },
        {
            'name': 'Mismatched upper bounds length',
            'test': lambda: ParticleCandidate.generate(2, [0, 0], [1]),
            'expected_error': ValueError
        },
        
        # Bounds relationship validation
        {
            'name': 'Lower bounds equal to upper bounds',
            'test': lambda: ParticleCandidate.generate(2, [5, 5], [5, 5]),
            'expected_error': ValueError
        },
        {
            'name': 'Lower bounds greater than upper bounds',
            'test': lambda: ParticleCandidate.generate(2, [10, 8], [5, 3]),
            'expected_error': ValueError
        },
        
        # Constructor validation tests  
        {
            'name': 'Invalid candidate array type',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], "not_array", [1, 1]),
            'expected_error': TypeError
        },
        {
            'name': 'Invalid velocity array type',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], [5, 5], "not_array"),
            'expected_error': TypeError
        },
        {
            'name': 'Candidate array wrong size',
            'test': lambda: ParticleCandidate(3, [0, 0, 0], [10, 10, 10], [5, 5], [1, 1, 1]),
            'expected_error': ValueError
        },
        {
            'name': 'Velocity array wrong size',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], [5, 5], [1, 1, 1]),
            'expected_error': ValueError
        },
        {
            'name': 'Candidate position outside bounds (below)',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], [-1, 5], [1, 1]),
            'expected_error': ValueError
        },
        {
            'name': 'Candidate position outside bounds (above)',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], [5, 15], [1, 1]),
            'expected_error': ValueError
        },
        
        # PSO weight validation tests
        {
            'name': 'Weights sum too high',
            'test': lambda: ParticleCandidate.generate(2, [0, 0], [10, 10], wl=0.5, wn=0.5, wg=0.5),
            'expected_error': ValueError
        },
        {
            'name': 'Weights sum too low',
            'test': lambda: ParticleCandidate.generate(2, [0, 0], [10, 10], wl=0.2, wn=0.2, wg=0.2),
            'expected_error': ValueError
        },
        
        # Array content validation tests
        {
            'name': 'Non-numeric candidate values',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], ["a", "b"], [1, 1]),
            'expected_error': ValueError
        },
        {
            'name': 'Non-numeric velocity values',
            'test': lambda: ParticleCandidate(2, [0, 0], [10, 10], [5, 5], ["x", "y"]),
            'expected_error': ValueError
        }
    ]
    
    passed_tests = 0
    total_tests = len(validation_tests)
    
    for test_case in validation_tests:
        try:
            test_case['test']()
            print(f"   FAILED: {test_case['name']} - Should have raised {test_case['expected_error'].__name__}")
        except test_case['expected_error']:
            print(f"   PASSED: {test_case['name']} - Correctly caught {test_case['expected_error'].__name__}")
            passed_tests += 1
        except Exception as e:
            print(f"   FAILED: {test_case['name']} - Unexpected error: {type(e).__name__}: {e}")
    
    print(f"   Validation tests: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

def test_pso_algorithm_properties():
    """Test PSO-specific algorithm properties"""
    
    print("\nTest 6: Testing PSO algorithm properties...")
    
    # Test that different particles have different random initializations
    particles = [ParticleCandidate.generate(3, [0, 0, 0], [10, 10, 10]) for _ in range(5)]
    positions = [p.candidate for p in particles]
    
    # Check that not all particles have identical positions (very unlikely with random generation)
    all_same = all(np.allclose(positions[0], pos) for pos in positions[1:])
    assert not all_same, "All particles have identical positions - randomization may be broken"
    
    # Test velocity update formula components
    p1 = ParticleCandidate.generate(2, [0, 0], [10, 10])
    local_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
    neighborhood_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
    global_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
    
    # Multiple velocity updates should give different results due to random components
    p2 = p1.recombine(local_best, neighborhood_best, global_best)
    p3 = p1.recombine(local_best, neighborhood_best, global_best)
    
    # Due to random factors rl, rn, rg, velocities should be different
    velocities_different = not np.allclose(p2.velocity, p3.velocity, rtol=1e-10)
    assert velocities_different, "Velocity updates are deterministic - random factors may be missing"
    
    print("   PASSED: PSO algorithm properties")

def run_all_tests():
    """Run comprehensive test suite"""
    
    print("ParticleCandidate Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        # Run core functionality tests
        particle = test_particle_creation()
        moved_particle = test_particle_movement(particle)
        updated_particle = test_velocity_update(moved_particle)
        test_boundary_handling()
        
        # Run validation tests
        validation_passed = test_comprehensive_validation()
        
        # Run PSO-specific tests
        test_pso_algorithm_properties()
        
        print("\n" + "=" * 50)
        if validation_passed:
            print("SUCCESS: All tests passed! ParticleCandidate implementation is robust and correct.")
        else:
            print("WARNING: Some validation tests failed. Check implementation.")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def demo_particle_behavior():
    """Demonstration of particle behavior over multiple steps"""
    
    print("\n" + "=" * 50)
    print("Particle Behavior Demonstration")
    print("=" * 50)
    
    # Create a particle
    particle = ParticleCandidate.generate(2, [0, 0], [10, 10])
    print(f"Initial particle: {particle}")
    
    # Show several steps of PSO behavior
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Create some random "best" positions
        local_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
        neighborhood_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
        global_best = ParticleCandidate.generate(2, [0, 0], [10, 10])
        
        # Update velocity (recombine)
        particle = particle.recombine(local_best, neighborhood_best, global_best)
        print(f"After velocity update: velocity = {particle.velocity}")
        
        # Move (mutate)
        particle = particle.mutate()
        print(f"After movement: position = {particle.candidate}")

if __name__ == "__main__":
    # Run the comprehensive test suite
    success = run_all_tests()
    
    if success:
        # If tests pass, show behavior demonstration
        demo_particle_behavior()
    else:
        print("\nFix the errors above before proceeding to ParticleSwarmOptimizer implementation.")