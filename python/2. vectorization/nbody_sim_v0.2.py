import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from datetime import datetime

# Simulation parameters
N = 1000              # Number of particles
DT = 0.01             # Time step
STEPS = 1000          # Number of simulation steps
G = 6.67430e-11       # Gravitational constant
SOFTENING = 1e9       # Softening factor to prevent singularities

class PerformanceTracker:
    """Track timing data for different functions"""
    def __init__(self, filename='performance_data.csv'):
        self.filename = filename
        self.timings = []
        
    def record(self, function_name, step, duration, method='nested_loops'):
        """Record a timing measurement"""
        self.timings.append({
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'function': function_name,
            'step': step,
            'duration_ms': duration * 1000,  # Convert to milliseconds
            'n_particles': N
        })
    
    def save(self):
        """Save all timings to CSV"""
        if not self.timings:
            return
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.timings[0].keys())
            writer.writeheader()
            writer.writerows(self.timings)
        
        print(f"\nPerformance data saved to {self.filename}")
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary of timing data"""
        if not self.timings:
            return
        
        # Group by method and function
        methods = {}
        for timing in self.timings:
            method = timing['method']
            func = timing['function']
            key = f"{method}_{func}"
            
            if key not in methods:
                methods[key] = []
            methods[key].append(timing['duration_ms'])
        
        print("\n=== Performance Summary ===")
        for key, durations in methods.items():
            avg_time = np.mean(durations)
            total_time = np.sum(durations)
            print(f"{key}: avg={avg_time:.3f}ms, total={total_time:.3f}ms")


class NBodySimulation:
    def __init__(self, n_particles, tracker, method='nested_loops'):
        self.n = n_particles
        self.tracker = tracker
        self.method = method
        
        # Initialize positions (3D coordinates)
        np.random.seed(42)  # For reproducibility
        self.positions = (np.random.rand(n_particles, 3) - 0.5) * 1e10
        
        # Initialize velocities
        self.velocities = (np.random.rand(n_particles, 3) - 0.5) * 1e4
        
        # Initialize masses
        self.masses = 1e20 + np.random.rand(n_particles) * 1e21
        
        # Store positions for visualization
        self.position_history = []
    
    def compute_forces_nested(self):
        """Calculate forces using nested loops - O(n^2) complexity"""
        forces = np.zeros((self.n, 3))
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Vector from particle i to particle j
                    r_vec = self.positions[j] - self.positions[i]
                    
                    # Distance with softening
                    r_mag = np.sqrt(np.sum(r_vec**2) + SOFTENING**2)
                    
                    # Force magnitude: F = G * m1 * m2 / r^2
                    force_mag = G * self.masses[i] * self.masses[j] / r_mag**3
                    
                    forces[i] += force_mag * r_vec
        
        return forces
    
    def compute_forces_vectorized(self):
        """Calculate forces using NumPy vectorization - much faster!"""
        # Compute all pairwise displacement vectors at once
        # positions shape: (n, 3)
        # We want: r_ij = positions[j] - positions[i] for all i,j
        
        # Reshape to enable broadcasting: (n, 1, 3) - (1, n, 3) = (n, n, 3)
        r_vec = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        
        # Calculate distances: shape (n, n)
        r_mag = np.sqrt(np.sum(r_vec**2, axis=2) + SOFTENING**2)
        
        # Avoid division by zero on diagonal (where i==j)
        np.fill_diagonal(r_mag, 1.0)
        
        # Calculate force magnitudes: (n, n)
        # Broadcasting: masses[:, np.newaxis] is (n, 1), masses[np.newaxis, :] is (1, n)
        force_mag = G * self.masses[:, np.newaxis] * self.masses[np.newaxis, :] / r_mag**3
        
        # Zero out self-interaction (diagonal)
        np.fill_diagonal(force_mag, 0.0)
        
        # Calculate force vectors: (n, n, 3)
        force_vectors = force_mag[:, :, np.newaxis] * r_vec
        
        # Sum forces on each particle: (n, 3)
        forces = np.sum(force_vectors, axis=1)
        
        return forces
    
    def compute_forces(self):
        """Wrapper that calls the appropriate force computation method"""
        if self.method == 'nested_loops':
            return self.compute_forces_nested()
        else:
            return self.compute_forces_vectorized()
    
    def update(self, dt, step):
        """Update velocities and positions"""
        # Time force computation
        start = time.time()
        forces = self.compute_forces()
        force_time = time.time() - start
        self.tracker.record('compute_forces', step, force_time, self.method)
        
        # Time velocity update
        start = time.time()
        accelerations = forces / self.masses[:, np.newaxis]
        self.velocities += accelerations * dt
        velocity_time = time.time() - start
        self.tracker.record('update_velocities', step, velocity_time, self.method)
        
        # Time position update
        start = time.time()
        self.positions += self.velocities * dt
        position_time = time.time() - start
        self.tracker.record('update_positions', step, position_time, self.method)
    
    def calculate_energy(self):
        """Calculate total kinetic energy (for verification)"""
        v_squared = np.sum(self.velocities**2, axis=1)
        kinetic_energy = 0.5 * np.sum(self.masses * v_squared)
        return kinetic_energy
    
    def run_simulation(self, steps, dt, save_interval=100):
        """Run the simulation for a given number of steps"""
        print(f"\nN-Body Simulation [{self.method}]: {self.n} particles, {steps} steps")
        print(f"Initial energy: {self.calculate_energy():.6e} J\n")
        
        # Start timing
        start_time = time.time()
        
        for step in range(steps):
            self.update(dt, step)
            
            # Print progress and energy
            if step % save_interval == 0:
                energy = self.calculate_energy()
                elapsed = time.time() - start_time
                print(f"Step {step:4d}: Energy = {energy:.6e} J | "
                      f"Time = {elapsed:.2f}s")
                
                # Save position snapshot for visualization
                if step % (save_interval * 2) == 0:  # Save less frequently
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
        ax.set_title(f'N-Body Simulation - {self.method} - Step {step_index}')
        
        plt.tight_layout()
        filename = f'nbody_{self.method}_step_{step_index}.png'
        plt.savefig(filename, dpi=150)
        print(f"Saved visualization to {filename}")
        plt.close()


def compare_methods():
    """Run both methods and compare performance"""
    print("="*60)
    print("N-BODY SIMULATION PERFORMANCE COMPARISON")
    print("="*60)
    
    tracker = PerformanceTracker()
    
    # Run nested loops version
    print("\n[1/2] Running NESTED LOOPS version...")
    sim1 = NBodySimulation(N, tracker, method='nested_loops')
    time1 = sim1.run_simulation(STEPS, DT)
    
    # Run vectorized version
    print("\n[2/2] Running VECTORIZED version...")
    sim2 = NBodySimulation(N, tracker, method='vectorized')
    time2 = sim2.run_simulation(STEPS, DT)
    
    # Save all timing data
    tracker.save()
    
    # Print comparison
    speedup = time1 / time2
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Nested loops:  {time1:.3f} seconds")
    print(f"Vectorized:    {time2:.3f} seconds")
    print(f"Speedup:       {speedup:.2f}x faster")
    print("="*60)
    
    # Visualize final states
    print("\nGenerating visualizations...")
    sim1.visualize_snapshot(step_index=-1)
    sim2.visualize_snapshot(step_index=-1)


def main():
    compare_methods()


if __name__ == "__main__":
    main()
