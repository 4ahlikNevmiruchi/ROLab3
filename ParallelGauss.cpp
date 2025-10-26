#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>    // For timing and random seed
#include <iomanip>   // For std::setw, std::setprecision
#include <random>    // For modern random number generation
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For exceptions
#include <mpi.h>

/**
 * @struct Colors
 * @brief Holds ANSI escape codes for colorful terminal output.
 */
struct Colors {
    static const char* RESET;
    static const char* RED;
    static const char* GREEN;
    static const char* YELLOW;
    static const char* CYAN;
    static const char* BOLD;
};
const char* Colors::RESET = "\033[0m";
const char* Colors::RED = "\033[31m";
const char* Colors::GREEN = "\033[32m";
const char* Colors::YELLOW = "\033[33m";
const char* Colors::CYAN = "\033[36m";
const char* Colors::BOLD = "\033[1m";

/**
 * @class ParallelGauss
 * @brief Encapsulates the state and logic for the parallel Gauss elimination.
 */
class ParallelGauss {
private:
    // MPI state
    int procRank;
    int procNum;

    // Problem size
    int size;   // Total size of the matrix (N)
    int rowNum; // Number of rows managed by this process

    // Data buffers
    // --- Root Process (procRank == 0) Only ---
    std::vector<double> pMatrix; // The full NxN matrix A
    std::vector<double> pVector; // The full Nx1 vector b
    std::vector<double> pResult; // The full Nx1 result vector x

    // --- All Processes ---
    std::vector<double> pProcRows;    // This process's horizontal stripe of A
    std::vector<double> pProcVector;  // This process's segment of b
    std::vector<double> pProcResult;  // This process's segment of the calculated x

    // MPI data distribution helpers (all processes)
    std::vector<int> pProcInd; // Global start index of rows for each process
    std::vector<int> pProcNum; // Number of rows for each process

    // Algorithm state (all processes)
    std::vector<int> pParallelPivotPos; // pParallelPivotPos[i] = global row index that was pivot for iter i
    std::vector<int> pProcPivotIter;    // pProcPivotIter[i] = iteration number where local row i was pivot (-1 if never)

    // C++11 Random Number Generator
    std::mt19937 randEngine;

public:
    /**
     * @brief Constructor: Initializes MPI state and seeds the random generator.
     */
    ParallelGauss(int rank, int num) : procRank(rank), procNum(num), size(0), rowNum(0) {
        // Seed the random engine uniquely for each process
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + procRank;
        randEngine.seed(seed);

        // Set output to be unbuffered on the root for cleaner logging
        if (procRank == 0) {
            std::cout.setf(std::ios::unitbuf);
        }
    }

    /**
     * @brief Main function to orchestrate the entire solution process.
     */
    void Run() {
        ProcessInitialization(); // Get size, bcast, and resize all vectors

        if (procRank == 0) {
            std::cout << Colors::CYAN << "Initializing matrix and vector..." << Colors::RESET << std::endl;

            // Choose one initialization method:

            // Use this for simple validation (result is all 1s)
            // DummyDataInitialization();

            // Use this for performance testing (NOW FIXED)
            RandomDataInitialization();
        }

        // --- Start Timer ---
        auto start = std::chrono::high_resolution_clock::now();

        DataDistribution();
        ParallelGaussianElimination();
        ParallelBackSubstitution();
        ResultCollection();

        // --- Stop Timer ---
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // --- Output Results ---
        if (procRank == 0) {
            std::cout << Colors::BOLD << Colors::GREEN << "\nParallel computation finished." << Colors::RESET << std::endl;
            std::cout << Colors::BOLD << "Result Vector (x_0, x_1, ...):" << Colors::RESET << std::endl;
            PrintResultVector(pResult);

            std::cout << Colors::BOLD << "\n\nTime of execution: " << Colors::CYAN
                      << duration.count() << " seconds" << Colors::RESET << std::endl;

            // User prompt to test the result
            char choice = 'n';
            std::cout << Colors::YELLOW << "\nDo you want to validate the result (A*x = b)? (y/n): \n" << Colors::RESET;
            std::cin >> choice;

            if (choice == 'y' || choice == 'Y') {
                TestResult();
            } else {
                std::cout << Colors::CYAN << "Skipping result validation." << Colors::RESET << std::endl;
            }
        }
    }

private:
    /**
     * @brief Gets matrix size from user, broadcasts it, and resizes all vectors.
     */
    void ProcessInitialization() {
        if (procRank == 0) {
            do {
                std::cout << Colors::YELLOW << "\nEnter the size of the (NxN) matrix (N): \n" << Colors::RESET;
                std::cin >> size;
                if (std::cin.fail()) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << Colors::RED << "Invalid input. Please enter an integer." << Colors::RESET << std::endl;
                    size = 0;
                } else if (size < procNum) {
                    std::cout << Colors::RED << "Size must be greater than or equal to the number of processes ("
                              << procNum << ")!" << Colors::RESET << std::endl;
                }
            } while (size < procNum);
        }

        // Broadcast the valid size to all processes
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // --- All Processes: Calculate row distribution and resize vectors ---

        // Resize distribution helpers
        pProcInd.resize(procNum);
        pProcNum.resize(procNum);

        // Calculate workload distribution (handles non-divisible sizes)
        int restRows = size;
        pProcInd[0] = 0;
        pProcNum[0] = size / procNum;
        restRows -= pProcNum[0];

        for (int i = 1; i < procNum; i++) {
            pProcNum[i] = restRows / (procNum - i);
            restRows -= pProcNum[i];
            pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
        }

        rowNum = pProcNum[procRank]; // Store this process's row count

        // Resize process-local vectors
        pProcRows.resize(rowNum * size);
        pProcVector.resize(rowNum);
        pProcResult.resize(rowNum);

        // Resize algorithm state vectors
        pParallelPivotPos.resize(size);
        pProcPivotIter.resize(rowNum, -1); // Initialize all to -1 (not pivoted)

        // Resize root-only vectors
        if (procRank == 0) {
            pMatrix.resize(size * size);
            pVector.resize(size);
            pResult.resize(size);
        }
    }

    /**
     * @brief Fills matrix/vector with 1s (for dummy test). Root only.
     */
    void DummyDataInitialization() {
        for (int i = 0; i < size; i++) {
            pVector[i] = i + 1;
            for (int j = 0; j < size; j++) {
                pMatrix[i * size + j] = (j <= i) ? 1.0 : 0.0;
            }
        }
    }

    /**
     * @brief Fills matrix/vector with random, diagonally dominant values. Root only.
     */
    void RandomDataInitialization() {
        // ---------------------
        // --- FIX IS HERE ---
        // ---------------------
        // Create a dense, diagonally-dominant matrix.
        // This is non-singular and numerically stable for pivoting.
        std::uniform_real_distribution<double> dist(1.0, 10.0); // Random values for matrix
        std::uniform_real_distribution<double> v_dist(1.0, 1000.0); // Random values for vector

        for (int i = 0; i < size; i++) {
            double off_diagonal_sum = 0.0;
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    double val = dist(randEngine);
                    pMatrix[i * size + j] = val;
                    off_diagonal_sum += std::fabs(val);
                }
            }
            // Set the diagonal element to be larger than the sum of all others
            // This guarantees the matrix is non-singular.
            pMatrix[i * size + i] = off_diagonal_sum + dist(randEngine); // Add another random val

            // Fill the vector
            pVector[i] = v_dist(randEngine);
        }
        // ---------------------
        // --- END OF FIX ---
        // ---------------------
    }

    /**
     * @brief Distributes pMatrix and pVector from root to all processes.
     */
    void DataDistribution() {
        // --- Scatter Matrix A ---
        // We need temporary send/index counts *for matrix elements*
        std::vector<int> pSendNum(procNum);
        std::vector<int> pSendInd(procNum);

        for (int i = 0; i < procNum; i++) {
            pSendNum[i] = pProcNum[i] * size; // num_rows * num_cols
            pSendInd[i] = pProcInd[i] * size; // start_row * num_cols
        }

        MPI_Scatterv(
            pMatrix.data(),    // Send buffer (root only)
            pSendNum.data(),   // Send counts for each process
            pSendInd.data(),   // Displacements for each process
            MPI_DOUBLE,
            pProcRows.data(),  // Receive buffer
            pSendNum[procRank],// Receive count for this process
            MPI_DOUBLE,
            0,                 // Root process
            MPI_COMM_WORLD
        );

        // --- Scatter Vector b ---
        // We can reuse pProcNum and pProcInd directly
        MPI_Scatterv(
            pVector.data(),    // Send buffer (root only)
            pProcNum.data(),   // Send counts (num_rows for each)
            pProcInd.data(),   // Displacements (start_row for each)
            MPI_DOUBLE,
            pProcVector.data(),// Receive buffer
            pProcNum[procRank],// Receive count
            MPI_DOUBLE,
            0,                 // Root
            MPI_COMM_WORLD
        );
    }

    /**
     * @brief Performs parallel forward elimination (Gaussian elimination).
     */
    void ParallelGaussianElimination() {
        // This vector holds the pivot row + pivot b-value for broadcasting
        std::vector<double> pPivotRow(size + 1);

        // Struct for MPI_MAXLOC: finds max value and rank of process that has it
        struct {
            double value;
            int rank;
        } procPivot, globalPivot;

        // Iterate through each column to eliminate it
        for (int i = 0; i < size; i++) {

            // 1. Find local pivot row
            double maxVal = -1.0;
            int pivotPos = -1;

            for (int j = 0; j < rowNum; j++) {
                if (pProcPivotIter[j] == -1) { // If this row hasn't been a pivot yet
                    double absVal = std::fabs(pProcRows[j * size + i]);
                    if (absVal > maxVal) {
                        maxVal = absVal;
                        pivotPos = j; // Local index of the pivot row
                    }
                }
            }

            procPivot.value = maxVal;
            procPivot.rank = procRank;

            // 2. Find global pivot row
            MPI_Allreduce(
                &procPivot,     // Send buffer
                &globalPivot,   // Receive buffer
                1,              // Count
                MPI_DOUBLE_INT, // Custom MPI datatype
                MPI_MAXLOC,     // Operation
                MPI_COMM_WORLD
            );

            // 3. Check for singular matrix
            if (globalPivot.value < 1.e-9) {
                if (procRank == 0) {
                     std::cerr << Colors::BOLD << Colors::RED
                               << "\nError: Matrix is singular or numerically unstable."
                               << " Pivot value is " << globalPivot.value
                               << " at iteration " << i << Colors::RESET << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1); // Stop all processes
            }


            // 4. Store pivot information
            if (procRank == globalPivot.rank) {
                pProcPivotIter[pivotPos] = i; // Mark local row as used in iter i
                pParallelPivotPos[i] = pProcInd[procRank] + pivotPos; // Store global row index
            }

            // 5. Broadcast the global pivot row index to all processes
            MPI_Bcast(
                &pParallelPivotPos[i],
                1,
                MPI_INT,
                globalPivot.rank,
                MPI_COMM_WORLD
            );

            // 6. Broadcast the pivot row data
            if (procRank == globalPivot.rank) {
                // Fill the broadcast buffer
                for (int j = 0; j < size; j++) {
                    pPivotRow[j] = pProcRows[pivotPos * size + j];
                }
                pPivotRow[size] = pProcVector[pivotPos]; // Append the b-value
            }

            MPI_Bcast(
                pPivotRow.data(),
                size + 1,
                MPI_DOUBLE,
                globalPivot.rank,
                MPI_COMM_WORLD
            );

            // 7. Perform column elimination on all local rows
            ParallelEliminateColumns(pPivotRow.data(), i);
        }
    }

    /**
     * @brief Helper to perform elimination step on local rows.
     */
    void ParallelEliminateColumns(const double* pPivotRow, int iter) {
        double pivotValue = pPivotRow[iter];

        for (int i = 0; i < rowNum; i++) {
            if (pProcPivotIter[i] == -1) { // If not a pivot row
                double pivotFactor = pProcRows[i * size + iter] / pivotValue;

                for (int j = iter; j < size; j++) {
                    pProcRows[i * size + j] -= pivotFactor * pPivotRow[j];
                }
                pProcVector[i] -= pivotFactor * pPivotRow[size]; // Update b-vector part
            }
        }
    }

    /**
     * @brief Performs parallel back substitution.
     */
    void ParallelBackSubstitution() {
        double iterResult; // The calculated value for x_i

        // Iterate backwards from the last unknown
        for (int i = size - 1; i >= 0; i--) {

            // 1. Find which process has the pivot row for this iteration
            int globalRowIndex = pParallelPivotPos[i];
            int iterProcRank = -1; // Rank of process holding the pivot row
            int iterPivotPos = -1; // Local index of pivot row on that process

            FindBackPivotRow(globalRowIndex, iterProcRank, iterPivotPos);

            // 2. The process with the pivot row calculates the result for x_i
            if (procRank == iterProcRank) {
                iterResult = pProcVector[iterPivotPos] / pProcRows[iterPivotPos * size + i];
                pProcResult[iterPivotPos] = iterResult;
            }

            // 3. Broadcast the result x_i to all processes
            MPI_Bcast(
                &iterResult,
                1,
                MPI_DOUBLE,
                iterProcRank,
                MPI_COMM_WORLD
            );

            // 4. Update the right-hand side (vector b) for all remaining rows
            for (int j = 0; j < rowNum; j++) {
                if (pProcPivotIter[j] < i) { // If this row solves for an earlier unknown
                    double val = pProcRows[j * size + i] * iterResult;
                    pProcVector[j] -= val;
                }
            }
        }
    }

    /**
     * @brief Finds which process holds a global row and its local index.
     */
    void FindBackPivotRow(int rowIndex, int &iterProcRank, int &iterPivotPos) {
        for (int i = 0; i < procNum - 1; i++) {
            if (rowIndex >= pProcInd[i] && rowIndex < pProcInd[i + 1]) {
                iterProcRank = i;
                break;
            }
        }
        if (iterProcRank == -1) {
            iterProcRank = procNum - 1; // It must be on the last process
        }
        iterPivotPos = rowIndex - pProcInd[iterProcRank];
    }

    /**
     * @brief Gathers the distributed pProcResult into pResult on the root.
     */
    void ResultCollection() {
        MPI_Gatherv(
            pProcResult.data(), // Send buffer
            pProcNum[procRank], // Send count for this process
            MPI_DOUBLE,
            pResult.data(),     // Receive buffer (root only)
            pProcNum.data(),    // Receive counts from all
            pProcInd.data(),    // Displacements from all
            MPI_DOUBLE,
            0,                  // Root
            MPI_COMM_WORLD
        );
    }

    /**
     * @brief Validates the result by computing A*x and checking if it equals b.
     */
    void TestResult() {
        // This function only runs on the root process
        if (procRank != 0) return;

        std::cout << Colors::CYAN << "Validating result..." << Colors::RESET << std::endl;

        std::vector<double> pRightPartVector(size, 0.0);
        bool correct = true;
        double accuracy = 1.e-6; // Allowed floating point error

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // A[i][j] * x[j]
                // We must use pParallelPivotPos to get the correct x_j
                pRightPartVector[i] += pMatrix[i * size + j] * pResult[pParallelPivotPos[j]];
            }

            // Check if (A*x)[i] is close to b[i]
            if (std::fabs(pRightPartVector[i] - pVector[i]) > accuracy) {
                correct = false;
            }
        }

        if (correct) {
            std::cout << Colors::BOLD << Colors::GREEN
                      << "The result of the parallel Gauss algorithm is correct."
                      << Colors::RESET << std::endl;
        } else {
            std::cout << Colors::BOLD << Colors::RED
                      << "The result of the parallel Gauss algorithm is NOT correct. Check your code."
                      << Colors::RESET << std::endl;
        }
    }

    /**
     * @brief Prints a vector with nice formatting.
     */
    void PrintVector(const std::vector<double>& vec) {
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << std::setw(10) << vec[i];
            if ((i + 1) % 8 == 0 || i == vec.size() - 1) {
                std::cout << "\n";
            } else {
                std::cout << " ";
            }
        }
        std::cout << Colors::RESET;
    }

    /**
     * @brief Prints the result vector, re-ordering it correctly using the pivot map.
     */
    void PrintResultVector(const std::vector<double>& result) {
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < size; ++i) {
            // Print x_i, which is stored in result[pParallelPivotPos[i]]
            std::cout << std::setw(10) << result[pParallelPivotPos[i]];
            if ((i + 1) % 8 == 0 || i == size - 1) {
                std::cout << "\n";
            } else {
                std::cout << " ";
            }
        }
        std::cout << Colors::RESET;
    }
};


/**
 * @brief Main entry point
 */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int procRank, procNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    if (procRank == 0) {
        std::cout << Colors::BOLD << Colors::CYAN
                  << "Parallel Gauss Algorithm for Solving Linear Systems"
                  << Colors::RESET << std::endl;
        std::cout << "Running on " << Colors::YELLOW << procNum << " processes."
                  << Colors::RESET << std::endl;
    }

    try {
        ParallelGauss solver(procRank, procNum);
        solver.Run();
    } catch (const std::exception& e) {
        if (procRank == 0) {
            std::cerr << Colors::RED << "An exception occurred: " << e.what() << Colors::RESET << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (...) {
        if (procRank == 0) {
            std::cerr << Colors::RED << "An unknown exception occurred." << Colors::RESET << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}