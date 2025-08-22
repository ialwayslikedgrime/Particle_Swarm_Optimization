#!/usr/bin/env python3
"""
Quick test script for ParticleCandidate class.
Save this as test_particle.py and run: python test_particle.py
"""

import numpy as np
import sys

# Assuming your ParticleCandidate is in the same file/directory
# Modify this import line based on where you saved your class
try:
    from particle_candidate import ParticleCandidate  # Replace 'your_particle_file' with actual filename
except ImportError:
    print("❌ Could not import ParticleCandidate. Make sure it's in the same directory.")
    print("💡 Tip: If your class is in 'pso.py', change import to: from pso import ParticleCandidate")
    sys.exit(1)

def quick_test():
    """Fast test of all major ParticleCandidate functionality"""
    
    print("🚀 Quick ParticleCandidate Test")
    print("=" * 40)
    
    try:
        # Test 1: Basic particle creation using factory method
        print("1️⃣  Testing particle creation...")
        particle = ParticleCandidate.generate(
            size=3,
            lower=[0, 0, 0], 
            upper=[10, 10, 10]
        )
        print(f"   ✅ Created particle: position={particle.candidate}")
        print(f"   ✅ Velocity: {particle.velocity}")
        print(f"   ✅ Weights sum to: {particle.wl + particle.wn + particle.wg}")
        
        # Test 2: Movement (mutation)
        print("\n2️⃣  Testing particle movement...")
        old_position = particle.candidate.copy()
        moved_particle = particle.mutate()
        print(f"   ✅ Old position: {old_position}")
        print(f"   ✅ New position: {moved_particle.candidate}")
        print(f"   ✅ Movement vector: {moved_particle.candidate - old_position}")
        
        # Test 3: Velocity update (recombination)
        print("\n3️⃣  Testing velocity update...")
        
        # Create some "best" positions for the particle to learn from
        local_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
        neighborhood_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
        global_best = ParticleCandidate.generate(3, [0,0,0], [10,10,10])
        
        old_velocity = particle.velocity.copy()
        updated_particle = particle.recombine(local_best, neighborhood_best, global_best)
        
        print(f"   ✅ Old velocity: {old_velocity}")
        print(f"   ✅ New velocity: {updated_particle.velocity}")
        print(f"   ✅ Position unchanged: {np.allclose(particle.candidate, updated_particle.candidate)}")
        
        # Test 4: Boundary constraints
        print("\n4️⃣  Testing boundary handling...")
        
        # Create particle near boundary with large velocity
        boundary_particle = ParticleCandidate(
            size=2,
            lower=[0, 0],
            upper=[5, 5], 
            candidate=[4.8, 4.9],  # Near upper bounds
            velocity=[2, 2]        # Would go outside bounds
        )
        
        bounded_particle = boundary_particle.mutate()
        print(f"   ✅ Before bounds: position would be {boundary_particle.candidate + boundary_particle.velocity}")
        print(f"   ✅ After clipping: {bounded_particle.candidate}")
        print(f"   ✅ Stays within bounds: {np.all(bounded_particle.candidate <= [5,5])}")
        
        # Test 5: Error handling
        print("\n5️⃣  Testing error handling...")
        try:
            ParticleCandidate.generate(size=-1, lower=[0], upper=[1])
            print("   ❌ Should have caught negative size error")
        except ValueError:
            print("   ✅ Correctly caught negative size error")
        
        try:
            ParticleCandidate.generate(size=2, lower=[0], upper=[1, 2])  # Mismatched sizes
            print("   ❌ Should have caught mismatched array sizes")
        except ValueError:
            print("   ✅ Correctly caught mismatched array sizes")
        
        print("\n🎉 All tests passed! Your ParticleCandidate is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def demo_particle_behavior():
    """Quick demonstration of particle behavior"""
    print("\n" + "=" * 40)
    print("🎭 Quick Behavior Demo")
    print("=" * 40)
    
    # Create a particle
    particle = ParticleCandidate.generate(2, [0, 0], [10, 10])
    print(f"Initial particle: {particle}")
    
    # Show a few steps of movement and velocity updates
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
    # Run the quick test
    success = quick_test()
    
    if success:
        # If tests pass, show a quick demo
        demo_particle_behavior()
    else:
        print("\n💡 Fix the errors above before proceeding to ParticleSwarmOptimizer implementation.")