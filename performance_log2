============================================================
N-BODY SIMULATION PERFORMANCE COMPARISON
============================================================

[1/4] Running NESTED LOOPS version...

N-Body Simulation [nested_loops]: 1000 particles, 1000 steps
Initial energy: 7.312448e+30 J

Step    0: Energy = 7.312448e+30 J | Time = 9.60s
Step  100: Energy = 7.312448e+30 J | Time = 964.60s
Step  200: Energy = 7.312448e+30 J | Time = 1886.60s
Step  300: Energy = 7.312448e+30 J | Time = 2806.52s
Step  400: Energy = 7.312448e+30 J | Time = 3724.26s
Step  500: Energy = 7.312448e+30 J | Time = 4640.60s
Step  600: Energy = 7.312448e+30 J | Time = 5557.61s
Step  700: Energy = 7.312448e+30 J | Time = 6475.01s
Step  800: Energy = 7.312448e+30 J | Time = 7390.79s
Step  900: Energy = 7.312448e+30 J | Time = 8309.83s

Simulation complete!
Total time: 9230.555 seconds
Time per step: 9.230555 seconds
Final energy: 7.312448e+30 J
Performance: 1.082e+05 interactions/second

[2/4] Running VECTORIZED version...

N-Body Simulation [vectorized]: 1000 particles, 1000 steps
Initial energy: 7.312448e+30 J

Step    0: Energy = 7.312448e+30 J | Time = 0.09s
Step  100: Energy = 7.312448e+30 J | Time = 9.89s
Step  200: Energy = 7.312448e+30 J | Time = 19.75s
Step  300: Energy = 7.312448e+30 J | Time = 29.57s
Step  400: Energy = 7.312448e+30 J | Time = 39.52s
Step  500: Energy = 7.312448e+30 J | Time = 49.38s
Step  600: Energy = 7.312448e+30 J | Time = 59.21s
Step  700: Energy = 7.312448e+30 J | Time = 69.25s
Step  800: Energy = 7.312448e+30 J | Time = 79.22s
Step  900: Energy = 7.312448e+30 J | Time = 89.06s

Simulation complete!
Total time: 98.849 seconds
Time per step: 0.098849 seconds
Final energy: 7.312448e+30 J
Performance: 1.011e+07 interactions/second

[3/4] Running NUMBA JIT version...
(First run will be slow due to JIT compilation...)

N-Body Simulation [numba]: 1000 particles, 1000 steps
Initial energy: 7.312448e+30 J

Step    0: Energy = 7.312448e+30 J | Time = 1.35s
Step  100: Energy = 7.312448e+30 J | Time = 1.84s
Step  200: Energy = 7.312448e+30 J | Time = 2.33s
Step  300: Energy = 7.312448e+30 J | Time = 2.82s
Step  400: Energy = 7.312448e+30 J | Time = 3.33s
Step  500: Energy = 7.312448e+30 J | Time = 3.82s
Step  600: Energy = 7.312448e+30 J | Time = 4.31s
Step  700: Energy = 7.312448e+30 J | Time = 4.81s
Step  800: Energy = 7.312448e+30 J | Time = 5.31s
Step  900: Energy = 7.312448e+30 J | Time = 5.84s

Simulation complete!
Total time: 6.497 seconds
Time per step: 0.006497 seconds
Final energy: 7.312448e+30 J
Performance: 1.538e+08 interactions/second

[4/4] Running NUMBA PARALLEL version...

N-Body Simulation [numba_parallel]: 1000 particles, 1000 steps
Initial energy: 7.312448e+30 J

Step    0: Energy = 7.312448e+30 J | Time = 1.33s
Step  100: Energy = 7.312448e+30 J | Time = 1.45s
Step  200: Energy = 7.312448e+30 J | Time = 1.57s
Step  300: Energy = 7.312448e+30 J | Time = 1.69s
Step  400: Energy = 7.312448e+30 J | Time = 1.81s
Step  500: Energy = 7.312448e+30 J | Time = 1.92s
Step  600: Energy = 7.312448e+30 J | Time = 2.04s
Step  700: Energy = 7.312448e+30 J | Time = 2.20s
Step  800: Energy = 7.312448e+30 J | Time = 2.31s
Step  900: Energy = 7.312448e+30 J | Time = 2.43s

Simulation complete!
Total time: 2.553 seconds
Time per step: 0.002553 seconds
Final energy: 7.312448e+30 J
Performance: 3.913e+08 interactions/second

Performance data saved to performance_data.csv

=== Performance Summary ===
nested_loops_compute_forces: avg=9230.523ms, total=9230522.773ms
nested_loops_update_velocities: avg=0.012ms, total=11.985ms
nested_loops_update_positions: avg=0.000ms, total=0.000ms
vectorized_compute_forces: avg=98.761ms, total=98760.576ms
vectorized_update_velocities: avg=0.038ms, total=37.941ms
vectorized_update_positions: avg=0.005ms, total=5.059ms
numba_compute_forces: avg=6.462ms, total=6462.302ms
numba_update_velocities: avg=0.025ms, total=25.101ms
numba_update_positions: avg=0.001ms, total=1.001ms
numba_parallel_compute_forces: avg=2.421ms, total=2421.477ms
numba_parallel_update_velocities: avg=0.082ms, total=82.006ms
numba_parallel_update_positions: avg=0.016ms, total=16.208ms

============================================================
FINAL COMPARISON
============================================================
Nested loops:      9230.555 seconds  (baseline)
Vectorized:          98.849 seconds  ( 93.38x speedup)
Numba JIT:            6.497 seconds  (1420.72x speedup)
Numba Parallel:       2.553 seconds  (3615.92x speedup)
============================================================
