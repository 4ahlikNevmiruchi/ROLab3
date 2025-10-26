#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>    // For timing and random seed
#include <iomanip>   // For std::setw, std::setprecision
#include <random>    // For modern random number generation
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For exceptions
#include <algorithm> // For std::copy

// g++ -o serial_gauss SerialGauss.cpp -std=c++17 -O3
// ./serial_gauss

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

// UI Helper Functions

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

void PrintSubHeader(const std::string& title) {
    std::cout << "\n" << Colors::MAGENTA << Colors::BOLD << "--- " << title << " ---" << Colors::RESET << std::endl;
}

// C++11 Random Number Generator (for initialization)
std::mt19937 randEngine;

/**
 * @brief Fills matrix/vector with random, diagonally dominant values.
 */
void DiagonalDominantInitialization(double* pMatrix, double* pVector, int Size) {
    std::uniform_real_distribution<double> dist(1.0, 10.0);
    std::uniform_real_distribution<double> v_dist(1.0, 1000.0);

    for (int i = 0; i < Size; i++) {
        double off_diagonal_sum = 0.0;
        for (int j = 0; j < Size; j++) {
            if (i != j) {
                double val = dist(randEngine);
                pMatrix[i * Size + j] = val;
                off_diagonal_sum += std::fabs(val);
            }
        }
        pMatrix[i * Size + i] = off_diagonal_sum + dist(randEngine);
        pVector[i] = v_dist(randEngine);
    }
}

/**
 * @brief Allocates memory and initializes data
 */
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& Size) {
    // Setting the size
    do {
        std::cout << Colors::BOLD << Colors::YELLOW << "\nEnter the size of the (NxN) matrix (N): \n" << Colors::RESET;
        std::cin >> Size;
        if (std::cin.fail() || Size <= 0) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << Colors::RED << "Invalid input. Please enter a positive integer." << Colors::RESET << std::endl;
            Size = 0;
        }
    } while (Size <= 0);

    // Memory allocation
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];

    std::cout << "Initializing matrix and vector for size "
              << Colors::YELLOW << Size << "x" << Size << Colors::RESET
              << " using " << Colors::CYAN << "Diagonally Dominant" << Colors::RESET
              << " method..." << std::endl;

    DiagonalDominantInitialization(pMatrix, pVector, Size);
}

/**
 * @brief Prints the final result vector using the pivot map
 */
void PrintResultVector(double* pResult, int Size, int* pSerialPivotPos) {
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < Size; ++i) {
        // Print x_i, which is stored in result[pSerialPivotPos[i]]
        std::cout << Colors::YELLOW << std::setw(10) << pResult[pSerialPivotPos[i]];
        if ((i + 1) % 8 == 0 || i == Size - 1) {
            std::cout << "\n";
        } else {
            std::cout << " ";
        }
    }
    std::cout << Colors::RESET;
}

/**
 * @brief Finds the pivot row
 */
int FindPivotRow(double* pMatrix, int Size, int Iter, int* pSerialPivotIter) {
    int PivotRow = -1;
    double MaxValue = -1.0;

    for (int i = 0; i < Size; i++) {
        if ((pSerialPivotIter[i] == -1)) {
             double absVal = std::fabs(pMatrix[i * Size + Iter]);
            if (absVal > MaxValue) {
                PivotRow = i;
                MaxValue = absVal;
            }
        }
    }

    if (MaxValue < 1.e-9) {
        std::cerr << Colors::BOLD << Colors::RED
                  << "\nError: Matrix is singular or numerically unstable."
                  << " Pivot value is " << MaxValue
                  << " at iteration " << Iter << Colors::RESET << std::endl;
        throw std::runtime_error("Singular matrix");
    }

    return PivotRow;
}

/**
 * @brief Performs column elimination
 */
void SerialColumnElimination(double* pMatrix, double* pVector, int Pivot, int Iter, int Size, int* pSerialPivotIter) { // Added pSerialPivotIter
    double PivotValue = pMatrix[Pivot * Size + Iter];

    for (int i = 0; i < Size; i++) {
        if (pSerialPivotIter[i] == -1) {
            double PivotFactor = pMatrix[i * Size + Iter] / PivotValue;
            for (int j = Iter; j < Size; j++) {
                pMatrix[i * Size + j] -= PivotFactor * pMatrix[Pivot * Size + j];
            }
            pVector[i] -= PivotFactor * pVector[Pivot];
        }
    }
}

/**
 * @brief Performs Gaussian elimination
 */
void SerialGaussianElimination(double* pMatrix, double* pVector, int Size, int* pSerialPivotPos, int* pSerialPivotIter) { // Added pivot arrays
    int Iter;
    int PivotRow;
    for (Iter = 0; Iter < Size; Iter++) {
        PivotRow = FindPivotRow(pMatrix, Size, Iter, pSerialPivotIter);
        pSerialPivotPos[Iter] = PivotRow;
        pSerialPivotIter[PivotRow] = Iter;
        SerialColumnElimination(pMatrix, pVector, PivotRow, Iter, Size, pSerialPivotIter);
    }
}

/**
 * @brief Performs back substitution
 */
void SerialBackSubstitution(double* pMatrix, double* pVector, double* pResult, int Size, int* pSerialPivotPos) { // Added pSerialPivotPos
    int RowIndex;
    for (int i = Size - 1; i >= 0; i--) {
        RowIndex = pSerialPivotPos[i];
        double pivotValue = pMatrix[RowIndex * Size + i];
        if (std::fabs(pivotValue) < 1.e-9) {
             pResult[i] = 0.0; // Avoid division by zero
        } else {
             pResult[i] = pVector[RowIndex] / pivotValue;
        }

        for (int j = 0; j < i; j++) {
            // Find the global row index for loop variable j using the pivot map
            int RowIndexJ = pSerialPivotPos[j];
            // Update pVector for the row corresponding to j
            pVector[RowIndexJ] -= pMatrix[RowIndexJ * Size + i] * pResult[i];
            // Set the element in the matrix to 0 (though not strictly necessary)
            pMatrix[RowIndexJ * Size + i] = 0;
        }
    }
}


/**
 * @brief Main calculation function
 */
void SerialResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size, int* pSerialPivotPos, int* pSerialPivotIter) { // Added pivot arrays
    // Pivot arrays are now allocated in main

    // Initialize pSerialPivotIter
    for (int i = 0; i < Size; i++) {
        pSerialPivotIter[i] = -1;
    }

    // Gaussian elimination
    SerialGaussianElimination(pMatrix, pVector, Size, pSerialPivotPos, pSerialPivotIter);

    // Back substitution
    SerialBackSubstitution(pMatrix, pVector, pResult, Size, pSerialPivotPos);

    // Pivot arrays are now deallocated in main
}

/**
 * @brief Frees all allocated memory (including pivot arrays)
 */
void ProcessTermination(double* pMatrix, double* pVector, double* pResult, int* pSerialPivotPos, int* pSerialPivotIter) {
    delete[] pMatrix;
    delete[] pVector;
    delete[] pResult;
    delete[] pSerialPivotPos; // Deallocate here
    delete[] pSerialPivotIter; // Deallocate here
}

/**
 * @brief Validates the result by computing A*x and checking if it equals b.
 * Needs the ORIGINAL matrix and vector copies, plus the pivot map.
 */
void TestResult(double* pMatrixCopy, double* pVectorCopy, double* pResult, int Size, int* pSerialPivotPos) { // Added pSerialPivotPos
    PrintSubHeader("Step 3: Validation");
    std::cout << Colors::CYAN << "Validating result by computing (A * x_computed) and comparing to b..." << Colors::RESET << std::endl;

    bool correct = true;
    double accuracy = 1.e-4;

    std::vector<double> pRightPartVector(Size, 0.0); // Use vector for safety

    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            // Multiply original matrix row i by the correctly ordered result vector element j
            pRightPartVector[i] += pMatrixCopy[i * Size + j] * pResult[pSerialPivotPos[j]];
        }
        if (std::fabs(pRightPartVector[i] - pVectorCopy[i]) > accuracy) {
            correct = false;
            std::cout << Colors::RED << "  Mismatch at row " << i << ": Expected " << pVectorCopy[i]
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
 * @brief Main function
 */
int main() {
    double* pMatrix;
    double* pVector;
    double* pResult;
    int Size;
    int* pSerialPivotPos = nullptr; // Initialize to nullptr
    int* pSerialPivotIter = nullptr; // Initialize to nullptr
    double* pMatrixCopy = nullptr;
    double* pVectorCopy = nullptr;


    // Seed the global random engine
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    randEngine.seed(seed);
    std::cout.setf(std::ios::unitbuf);

    PrintHeader("Serial Gauss Algorithm Timer");

    try {
        // --- Step 1: Initialization ---
        PrintSubHeader("Step 1: Initialization");
        ProcessInitialization(pMatrix, pVector, pResult, Size);

        // Allocate pivot arrays here
        pSerialPivotPos = new int[Size];
        pSerialPivotIter = new int[Size];

        // We need copies for validation
        pMatrixCopy = new double[Size*Size];
        pVectorCopy = new double[Size];
        std::copy(pMatrix, pMatrix + (Size*Size), pMatrixCopy);
        std::copy(pVector, pVector + Size, pVectorCopy);


        // --- Step 2: Serial Computation ---
        PrintSubHeader("Step 2: Serial Computation");
        std::cout << "  - Starting serial calculation..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        // Pass pivot arrays to the calculation function
        SerialResultCalculation(pMatrix, pVector, pResult, Size, pSerialPivotPos, pSerialPivotIter);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "  - Calculation finished." << std::endl;

        if (Size <= 100) {
            std::cout << Colors::BOLD << "Result Vector (x_0, x_1, ...):" << Colors::RESET << std::endl;
            // Pass pivot array to the print function
            PrintResultVector(pResult, Size, pSerialPivotPos);
        } else {
            std::cout << Colors::BOLD << "Result Vector:" << Colors::RESET << std::endl;
            std::cout << Colors::CYAN << "(Vector is " << Size
                      << " elements long, too large to display)" << Colors::RESET << std::endl;
        }

        std::cout << Colors::BOLD << "\nTime of execution: " << Colors::CYAN
                  << std::fixed << std::setprecision(6)
                  << duration.count() << " seconds" << Colors::RESET << std::endl;


        // --- Step 3: Validation ---
        // Pass copies and pivot array to the test function
        TestResult(pMatrixCopy, pVectorCopy, pResult, Size, pSerialPivotPos);

        // --- Step 4: Cleanup ---
        PrintSubHeader("Step 4: Cleanup");
        // Pass all allocated arrays to termination function
        ProcessTermination(pMatrixCopy, pVectorCopy, pResult, pSerialPivotPos, pSerialPivotIter);
        // pMatrix and pVector were overwritten by the algorithm, but allocated separately
        delete[] pMatrix;
        delete[] pVector;
        std::cout << "Memory deallocated." << std::endl;


    } catch (const std::exception& e) {
        std::cerr << Colors::RED << "An exception occurred: " << e.what() << Colors::RESET << std::endl;
        // Need to clean up potentially allocated memory if error happened mid-way
        delete[] pMatrix; delete[] pVector; delete[] pResult;
        delete[] pSerialPivotPos; delete[] pSerialPivotIter;
        delete[] pMatrixCopy; delete[] pVectorCopy;
        return 1;
    } catch (...) {
        std::cerr << Colors::RED << "An unknown exception occurred." << Colors::RESET << std::endl;
        // Need to clean up potentially allocated memory
        delete[] pMatrix; delete[] pVector; delete[] pResult;
        delete[] pSerialPivotPos; delete[] pSerialPivotIter;
        delete[] pMatrixCopy; delete[] pVectorCopy;
        return 1;
    }

    return 0;
}