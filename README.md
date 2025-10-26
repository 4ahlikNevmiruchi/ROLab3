
| Matrix size | Serial algorithm |  2 ps Time | 2 ps Speed up |  4 ps Time | 4 ps Speed up |          8 ps Time | 8 ps Speed up |
|------------:|-----------------:|-----------:|--------------:|-----------:|--------------:|-------------------:|--------------:|
|          10 |         0.000002 |   0.000078 |         0.026 |   0.000160 |         0.013 |           0.000517 |         0.004 |
|         100 |         0.000093 |   0.000279 |         0.333 |   0.000478 |         0.195 |           0.000778 |         0.120 |
|       1 000 |         0.081901 |   0.061084 |         1.341 |   0.037670 |         2.174 |           0.028919 |         2.832 |
|       2 500 |         1.268733 |   0.855590 |         1.483 |   0.490301 |         2.588 |           0.277175 |         4.577 |
|       5 000 |        20.820184 |  12.579701 |         1.655 |   9.224463 |         2.257 |           8.159503 |         2.552 |
|       7 500 |        81.934472 |  58.850180 |         1.392 |  49.288562 |         1.662 |          50.279829 |         1.630 |
|      10 000 |       201.314107 | 151.575016 |         1.328 | 134.912463 |         1.492 |         141.239449 |         1.425 |

## Serial Bottlenecks in Parallel Gaussian Elimination

Even though we parallelize the main computation, several parts of the Gaussian elimination 
algorithm *must* happen sequentially or involve significant coordination, limiting the overall speedup:

* **Pivot Search Synchronization:** In each iteration ($N$ times total), 
all processes must compare their local maximum pivot candidates to find the single global maximum. 
This requires an `MPI_Allreduce` operation, forcing faster 
processes to **wait** for the slowest one before anyone can proceed.
* **Pivot Row Broadcast:** After the global pivot is found, the *one* process 
holding that row must broadcast it to *all* other processes. 
This is an `MPI_Bcast` operation. While other processes wait to receive the data, they cannot do useful computation.
* **Back Substitution Dependencies:** While some parts of back substitution can overlap, calculating 
each unknown $x_i$ often depends on the value of $x_{i+1}$ (which might be calculated on a different 
processor). This creates dependencies that require broadcasting the newly 
calculated $x_i$ and introduces waiting time.
* **Initial Data Distribution:** Scattering the initial matrix and vector 
(`MPI_Scatterv`) happens mostly sequentially from the root process.
* **Final Result Gathering:** Collecting the final result vector 
(`MPI_Gatherv`) also happens mostly sequentially towards the root process.

---

## Why Speedups Can Be Low (Diminishing Returns)

Low speedups, especially with more processors or larger matrices, 
are common for this algorithm due to **communication overhead**:

* **Communication vs. Computation:** The time spent sending data (pivots, rows) 
and synchronizing (`MPI_Allreduce`, `MPI_Bcast`) can become **larger** than the 
time saved by doing the math in parallel. This is especially true for:
    * **Small Matrices:** The math is so fast that the fixed cost of communication dominates.
    * **Large Matrices:** Sending large pivot rows ($N$ elements) takes significant time and network bandwidth.
* **Network Saturation:** With many processors communicating frequently, 
the network connecting them can become a bottleneck, slowing everyone down.
* **Amdahl's Law:** The serial portions (like the communication steps listed above) fundamentally limit 
the maximum possible speedup. No matter how many processors you add, the total time can 
never be faster than the time required for these serial parts.

In essence, parallel Gaussian elimination with pivoting is **communication-intensive**. 
While parallelism helps with the $O(N^3)$ computation, the $N$ rounds of expensive communication 
often prevent it from scaling linearly, leading to disappointing speedups.
