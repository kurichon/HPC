import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
N = 1000             # Number of particles
DT = 0.01             # Time step
STEPS = 1000          # Number of simulation steps
G = 6.67430e-11       # Gravitational constant
SOFTENING = 1e9       # Softening factor to prevent singularities

class NBodySimulation:
    def __init__(self, n_particles):
        self.n = n_particles
        
        # Initialize positions (3D coordinates)
        self.positions = (np.random.rand(n_particles, 3) - 0.5) * 1e10
        
        # Initialize velocities
        self.velocities = (np.random.rand(n_particles, 3) - 0.5) * 1e4
        
        # Initialize masses
        self.masses = 1e20 + np.random.rand(n_particles) * 1e21
        
        # Store positions for visualization
        self.position_history = []
    
    def compute_forces(self):
        """Calculate gravitational forces between all particles - O(n^2) complexity"""
        forces = np.zeros((self.n, 3))
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Vector from particle i to particle j
                    r_vec = self.positions[j] - self.positions[i]
                    
                    # Distance with softening
                    r_mag = np.sqrt(np.sum(r_vec**2) + SOFTENING**2)
                    
                    # Force magnitude: F = G * m1 * m2 / r^2
                    # Direction: normalized r_vec
                    force_mag = G * self.masses[i] * self.masses[j] / r_mag**3
                    
                    forces[i] += force_mag * r_vec
        
        return forces
    
    def update(self, dt):
        """Update velocities and positions"""
        # Compute all forces
        forces = self.compute_forces()
        
        # Update velocities: v = v + (F/m) * dt
        accelerations = forces / self.masses[:, np.newaxis]
        self.velocities += accelerations * dt
        
        # Update positions: x = x + v * dt
        self.positions += self.velocities * dt
    
    def calculate_energy(self):
        """Calculate total kinetic energy (for verification)"""
        v_squared = np.sum(self.velocities**2, axis=1)
        kinetic_energy = 0.5 * np.sum(self.masses * v_squared)
        return kinetic_energy
    
    def run_simulation(self, steps, dt, save_interval=100):
        """Run the simulation for a given number of steps"""
        print(f"N-Body Simulation: {self.n} particles, {steps} steps")
        print(f"Initial energy: {self.calculate_energy():.6e} J\n")
        
        # Start timing
        start_time = time.time()
        
        for step in range(steps):
            self.update(dt)
            
            # Print progress and energy
            if step % save_interval == 0:
                energy = self.calculate_energy()
                elapsed = time.time() - start_time
                print(f"Step {step:4d}: Energy = {energy:.6e} J | "
                      f"Time = {elapsed:.2f}s")
                
                # Save position snapshot for visualization
                self.position_history.append(self.positions.copy())
        
        # Final statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nSimulation complete!")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Time per step: {total_time/steps:.6f} seconds")
        print(f"Final energy: {self.calculate_energy():.6e} J")
        
        # Calculate performance metrics
        interactions = self.n * (self.n - 1) * steps
        print(f"Performance: {interactions/total_time:.3e} interactions/second")
        
        return total_time
    
    def visualize_snapshot(self, step_index=0):
        """Visualize a snapshot of particle positions"""
        if not self.position_history:
            positions = self.positions
        else:
            positions = self.position_history[step_index]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='blue', marker='o', s=1, alpha=0.6)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'N-Body Simulation - Snapshot at step {step_index}')
        
        plt.tight_layout()
        plt.savefig(f'nbody_snapshot_{step_index}.png', dpi=150)
        print(f"Saved visualization to nbody_snapshot_{step_index}.png")
        plt.show()


def main():
    # Create and run simulation
    sim = NBodySimulation(N)
    sim.run_simulation(STEPS, DT)
    
    # Visualize final state
    print("\nGenerating visualization...")
    sim.visualize_snapshot(step_index=-1)


if __name__ == "__main__":
    main()
