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
    static const char* MAGENTA;
    static const char* BOLD;
    static const char* WHITE_BOLD;
};
const char* Colors::RESET = "\033[0m";
const char* Colors::RED = "\033[31m";
const char* Colors::GREEN = "\033[32m";
const char* Colors::YELLOW = "\033[33m";
const char* Colors::CYAN = "\033[36m";
const char* Colors::MAGENTA = "\033[35m";
const char* Colors::BOLD = "\033[1m";
const char* Colors::WHITE_BOLD = "\033[1;37m";

// --- UI Helper Functions ---

/**
 * @brief Prints boxed header.
 */
void PrintHeader(const std::string& title) {
    std::cout << "\n" << Colors::WHITE_BOLD;
    std::cout << "╔";
    for (int i = 0; i < title.length() + 4; ++i) std::cout << "═";
    std::cout << "╗\n";
    std::cout << "║  " << Colors::CYAN << Colors::BOLD << title << Colors::WHITE_BOLD << "  ║\n";
    std::cout << "╚";
    for (int i = 0; i < title.length() + 4; ++i) std::cout << "═";
    std::cout << "╝" << Colors::RESET << std::endl;
}

/**
 * @brief Prints a formatted sub-header for steps.
 */
void PrintSubHeader(const std::string& title) {
    std::cout << "\n" << Colors::MAGENTA << Colors::BOLD << "--- " << title << " ---" << Colors::RESET << std::endl;
}

/**
 * @brief Prints the initial Matrix A and Vector b side-by-side.
 * Only called if size <= 20.
 */
void PrintMatrixVector(const std::vector<double>& matrix, const std::vector<double>& vec, int n) {
    PrintSubHeader("Initial System (A|b)");
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < n; ++i) {
        std::cout << Colors::CYAN << "║ "; // Start of row
        // Print matrix row
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << matrix[i * n + j];
        }
        std::cout << Colors::CYAN << " ║ " << Colors::YELLOW << std::setw(10) << vec[i]
                  << Colors::CYAN << " ║" << Colors::RESET << "\n";
    }
}


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

    // Problem generation
    int initChoice; // 1 = Dummy, 2 = Diagonal, 3 = Known Solution, 4 = Lab's Random

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
    ParallelGauss(int rank, int num) : procRank(rank), procNum(num), size(0), rowNum(0), initChoice(0) {
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
        // --- Step 1: Initialization ---
        if (procRank == 0) PrintSubHeader("Step 1: Initialization");
        ProcessInitialization(); // Gets size and initChoice

        if (procRank == 0) {
            std::string choiceName;
            if (initChoice == 1) {
                choiceName = "Dummy (Lab's simple test)";
                DummyDataInitialization();
            } else if (initChoice == 2) {
                choiceName = "Diagonally Dominant";
                DiagonalDominantInitialization();
            } else if (initChoice == 3) {
                choiceName = "Known Solution (Random x_true)";
                KnownSolutionInitialization();
            } else {
                choiceName = "Lab's Random Test (Shows failure)";
                LabRandomInitialization();
            }

            std::cout << "Initializing matrix and vector for size "
                      << Colors::YELLOW << size << "x" << size << Colors::RESET
                      << " using " << Colors::CYAN << choiceName << Colors::RESET
                      << " method..." << std::endl;

            // Print initial matrix if N <= 20 ***
            if (size <= 20) {
                PrintMatrixVector(pMatrix, pVector, size);
            } else {
                std::cout << Colors::CYAN << "(Matrix is " << size << "x" << size
                          << ", too large to display)" << Colors::RESET << std::endl;
            }
        }

        // --- Step 2: Parallel Computation ---
        if (procRank == 0) PrintSubHeader("Step 2: Parallel Computation");

        if (procRank == 0) std::cout << "  - Distributing data..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        DataDistribution();

        if (procRank == 0) std::cout << "  - Performing Gaussian elimination..." << std::endl;
        ParallelGaussianElimination();

        if (procRank == 0) std::cout << "  - Performing back substitution..." << std::endl;
        ParallelBackSubstitution();

        if (procRank == 0) std::cout << "  - Collecting results..." << std::endl;
        ResultCollection();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // --- Step 3: Results ---
        if (procRank == 0) {
            PrintSubHeader("Step 3: Results");

            // Print result vector if N <= 100 ***
            if (size <= 100) {
                std::cout << Colors::BOLD << "Result Vector (x_0, x_1, ...):" << Colors::RESET << std::endl;
                PrintResultVector(pResult);
            } else {
                std::cout << Colors::BOLD << "Result Vector:" << Colors::RESET << std::endl;
                std::cout << Colors::CYAN << "(Vector is " << size
                          << " elements long, too large to display)" << Colors::RESET << std::endl;
            }

            std::cout << Colors::BOLD << "\nTime of execution: " << Colors::CYAN
                      << std::fixed << std::setprecision(6)
                      << duration.count() << " seconds" << Colors::RESET << std::endl;

            // --- Step 4: Validation (Optional) ---
            char choice = 'n';
            std::cout << Colors::BOLD << Colors::YELLOW << "\nDo you want to validate the result (A*x = b)? (y/n): \n" << Colors::RESET;
            std::cin >> choice;

            if (choice == 'y' || choice == 'Y') {
                TestResult();
            } else {
                PrintSubHeader("Step 4: Validation");
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
                std::cout << Colors::BOLD << Colors::YELLOW << "\nEnter the size of the (NxN) matrix (N): \n" << Colors::RESET;
                std::cin >> size;
                if (std::cin.fail() || size <= 0) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << Colors::RED << "Invalid input. Please enter a positive integer." << Colors::RESET << std::endl;
                    size = 0;
                } else if (size < procNum) {
                    std::cout << Colors::RED << "Size must be greater than or equal to the number of processes ("
                              << procNum << ")!" << Colors::RESET << std::endl;
                }
            } while (size < procNum);

            // Prompt for initialization method
            do {
                std::cout << Colors::BOLD << Colors::YELLOW << "\nChoose initialization method:" << Colors::RESET << std::endl;
                std::cout << "  (1) " << Colors::CYAN << "Dummy Data" << Colors::RESET << " (Simple, non-random test)" << std::endl;
                std::cout << "  (2) " << Colors::CYAN << "Diagonally Dominant" << Colors::RESET << " (Stable random test)" << std::endl;
                std::cout << "  (3) " << Colors::CYAN << "Known Solution" << Colors::RESET << " (Dense random A, random x_true)" << std::endl;
                std::cout << "  (4) " << Colors::CYAN << "Random Test" << Colors::RESET << " (" << Colors::RED << "Shows singular matrix error" << Colors::RESET << ")" << std::endl; // <-- NEW
                std::cout << "Your choice (1-4): \n";
                std::cin >> initChoice;
                if (std::cin.fail() || initChoice < 1 || initChoice > 4) { // <-- UPDATED
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << Colors::RED << "Invalid choice. Please enter 1, 2, 3, or 4." << Colors::RESET << std::endl; // <-- UPDATED
                    initChoice = 0;
                }
            } while (initChoice == 0);
        }

        // Broadcast the valid size to all processes
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // Broadcast the init choice to all processes
        MPI_Bcast(&initChoice, 1, MPI_INT, 0, MPI_COMM_WORLD);


        // --- All Processes: Calculate row distribution and resize vectors ---
        pProcInd.resize(procNum);
        pProcNum.resize(procNum);

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

        pProcRows.resize(rowNum * size);
        pProcVector.resize(rowNum);
        pProcResult.resize(rowNum);
        pParallelPivotPos.resize(size);
        pProcPivotIter.resize(rowNum, -1); // Initialize all to -1 (not pivoted)

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
        // This is the simple test from the lab manual [cite: 254-268].
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
    void DiagonalDominantInitialization() {
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
            pMatrix[i * size + i] = off_diagonal_sum + dist(randEngine);
            pVector[i] = v_dist(randEngine);
        }
    }

    /**
     * @brief Fills A with random values, x_true with random values, and calculates b = A*x_true.
     */
    void KnownSolutionInitialization() {
        std::uniform_real_distribution<double> a_dist(1.0, 10.0); // For matrix A
        std::uniform_real_distribution<double> x_dist(1.0, 10.0); // For solution x_true

        // 1. Generate the random "ground-truth" solution
        std::vector<double> x_true(size);
        for (int i = 0; i < size; ++i) {
            x_true[i] = x_dist(randEngine);
        }

        // 2. Generate random matrix A and calculate b = A * x_true
        for (int i = 0; i < size; i++) {
            pVector[i] = 0.0; // Clear pVector to use as an accumulator
            for (int j = 0; j < size; j++) {
                double val = a_dist(randEngine);
                pMatrix[i * size + j] = val;
                // Calculate pVector[i] = row i of A * x_true
                pVector[i] += val * x_true[j];
            }
        }
    }

    /**
     * @brief Fills matrix/vector with the lab's original (flawed) random method.
     */
    void LabRandomInitialization() {
        // This is the flawed random test from the lab manual .
        // It creates a sparse, lower-triangular matrix.
        std::uniform_real_distribution<double> dist(0.0, 1.0); // rand() / double(1000)

        for (int i = 0; i < size; i++) {
            pVector[i] = dist(randEngine) * 1000.0; // [cite: 311]
            for (int j = 0; j < size; j++) {
                if (j <= i) {
                    pMatrix[i * size + j] = dist(randEngine) * 1000.0; // [cite: 313-314]
                } else {
                    pMatrix[i * size + j] = 0.0; // [cite: 315-316]
                }
            }
        }
    }

    /**
     * @brief Distributes pMatrix and pVector from root to all processes.
     */
    void DataDistribution() {
        std::vector<int> pSendNum(procNum);
        std::vector<int> pSendInd(procNum);

        for (int i = 0; i < procNum; i++) {
            pSendNum[i] = pProcNum[i] * size; // num_rows * num_cols
            pSendInd[i] = pProcInd[i] * size; // start_row * num_cols
        }

        MPI_Scatterv(
            pMatrix.data(), pSendNum.data(), pSendInd.data(), MPI_DOUBLE,
            pProcRows.data(), pSendNum[procRank], MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        MPI_Scatterv(
            pVector.data(), pProcNum.data(), pProcInd.data(), MPI_DOUBLE,
            pProcVector.data(), pProcNum[procRank], MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }

    /**
     * @brief Performs parallel forward elimination (Gaussian elimination).
     */
    void ParallelGaussianElimination() {
        std::vector<double> pPivotRow(size + 1);
        struct {
            double value;
            int rank;
        } procPivot, globalPivot;

        for (int i = 0; i < size; i++) {
            double maxVal = -1.0;
            int pivotPos = -1;

            for (int j = 0; j < rowNum; j++) {
                if (pProcPivotIter[j] == -1) {
                    double absVal = std::fabs(pProcRows[j * size + i]);
                    if (absVal > maxVal) {
                        maxVal = absVal;
                        pivotPos = j;
                    }
                }
            }

            procPivot.value = maxVal;
            procPivot.rank = procRank;

            MPI_Allreduce(&procPivot, &globalPivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

            // Special check for DummyData (Option 1) & Lab's Random (Option 4)
            // Both can have 0 pivots after the lower-triangular part is done.
            if ((initChoice == 1 || initChoice == 4) && globalPivot.value < 1.e-9) {
                // This is expected for the upper-right triangle of zeros.
                // We just need to mark a row as "used" and continue.

                // Find *any* unused row on this process
                if(pivotPos == -1 && procRank == globalPivot.rank) {
                     for(int j=0; j<rowNum; ++j) {
                        if(pProcPivotIter[j] == -1) {
                            pivotPos = j;
                            break;
                        }
                    }
                }

                if (procRank == globalPivot.rank) {
                    if (pivotPos != -1) { // This process has an unused row
                        pProcPivotIter[pivotPos] = i;
                        pParallelPivotPos[i] = pProcInd[procRank] + pivotPos;
                    } else {
                        // This process has no unused rows. Another process will.
                        // We must set a "valid" index, even if it's a repeat,
                        // so the Bcast doesn't fail.
                        pParallelPivotPos[i] = pProcInd[procRank]; // Just send something
                    }
                }
                // Broadcast the "chosen" pivot row, even if it's 0
                MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, globalPivot.rank, MPI_COMM_WORLD);
                // No elimination is needed, so just continue the loop
                continue;
            }

            // Standard check for non-dummy methods (Options 2 & 3)
            if (globalPivot.value < 1.e-9) {
                if (procRank == 0) {
                     std::cerr << Colors::BOLD << Colors::RED
                               << "\nError: Matrix is singular or numerically unstable."
                               << " Pivot value is " << globalPivot.value
                               << " at iteration " << i << Colors::RESET << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            if (procRank == globalPivot.rank) {
                pProcPivotIter[pivotPos] = i;
                pParallelPivotPos[i] = pProcInd[procRank] + pivotPos;
            }

            MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, globalPivot.rank, MPI_COMM_WORLD);

            if (procRank == globalPivot.rank) {
                for (int j = 0; j < size; j++) {
                    pPivotRow[j] = pProcRows[pivotPos * size + j];
                }
                pPivotRow[size] = pProcVector[pivotPos];
            }

            MPI_Bcast(pPivotRow.data(), size + 1, MPI_DOUBLE, globalPivot.rank, MPI_COMM_WORLD);

            ParallelEliminateColumns(pPivotRow.data(), i);
        }
    }

    /**
     * @brief Helper to perform elimination step on local rows.
     */
    void ParallelEliminateColumns(const double* pPivotRow, int iter) {
        double pivotValue = pPivotRow[iter];

        // This check is important for the DummyData/LabRandom cases
        if (std::fabs(pivotValue) < 1.e-9) {
            return; // No elimination possible
        }

        for (int i = 0; i < rowNum; i++) {
            if (pProcPivotIter[i] == -1) {
                double pivotFactor = pProcRows[i * size + iter] / pivotValue;

                for (int j = iter; j < size; j++) {
                    pProcRows[i * size + j] -= pivotFactor * pPivotRow[j];
                }
                pProcVector[i] -= pivotFactor * pPivotRow[size];
            }
        }
    }

    /**
     * @brief Performs parallel back substitution.
     */
    void ParallelBackSubstitution() {
        double iterResult;
        for (int i = size - 1; i >= 0; i--) {
            int globalRowIndex = pParallelPivotPos[i];
            int iterProcRank = -1;
            int iterPivotPos = -1;

            FindBackPivotRow(globalRowIndex, iterProcRank, iterPivotPos);

            if (procRank == iterProcRank) {
                double pivotValue = pProcRows[iterPivotPos * size + i];
                if (std::fabs(pivotValue) < 1.e-9) {
                    iterResult = 0.0; // This happens with DummyData/LabRandom
                } else {
                    iterResult = pProcVector[iterPivotPos] / pivotValue;
                }
                pProcResult[iterPivotPos] = iterResult;
            }

            MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);

            for (int j = 0; j < rowNum; j++) {
                if (pProcPivotIter[j] < i) {
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
            iterProcRank = procNum - 1;
        }
        iterPivotPos = rowIndex - pProcInd[iterProcRank];
    }

    /**
     * @brief Gathers the distributed pProcResult into pResult on the root.
     */
    void ResultCollection() {
        MPI_Gatherv(
            pProcResult.data(), pProcNum[procRank], MPI_DOUBLE,
            pResult.data(), pProcNum.data(), pProcInd.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }

    /**
     * @brief Validates the result by computing A*x and checking if it equals b.
     * This method is robust and works for all 4 initialization types.
     */
    void TestResult() {
        if (procRank != 0) return;

        PrintSubHeader("Step 4: Validation");
        std::cout << Colors::CYAN << "Validating result by computing (A * x_computed) and comparing to b..." << Colors::RESET << std::endl;

        std::vector<double> pRightPartVector(size, 0.0);
        bool correct = true;
        // Use a slightly more relaxed accuracy for larger/complex problems
        double accuracy = (initChoice == 1) ? 1.e-6 : 1.e-4;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                pRightPartVector[i] += pMatrix[i * size + j] * pResult[pParallelPivotPos[j]];
            }
            if (std::fabs(pRightPartVector[i] - pVector[i]) > accuracy) {
                correct = false;
                 std::cout << Colors::RED << "  Mismatch at row " << i << ": Expected " << pVector[i]
                           << ", Got " << pRightPartVector[i] << Colors::RESET << std::endl;
            }
        }

        if (correct) {
            std::cout << Colors::BOLD << Colors::GREEN
                      << "VALIDATION SUCCESSFUL: The result is correct."
                      << Colors::RESET << std::endl;
        } else {
            std::cout << Colors::BOLD << Colors::RED
                      << "VALIDATION FAILED: The result is NOT correct."
                      << Colors::RESET << std::endl;
        }
    }

    /**
     * @brief Prints the final result vector, re-ordering it correctly.
     */
    void PrintResultVector(const std::vector<double>& result) {
        std::cout << std::fixed << std::setprecision(4);
        for (int i = 0; i < size; ++i) {
            // Print x_i, which is stored in result[pParallelPivotPos[i]]
            std::cout << Colors::YELLOW << std::setw(10) << result[pParallelPivotPos[i]];
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
        PrintHeader("Parallel Gauss Algorithm Solver");
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