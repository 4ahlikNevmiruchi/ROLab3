import java.util.Scanner;
import java.util.Random;
import java.lang.Math; // For fabs (Math.abs)

/**
 * A Java implementation of the serial Gauss algorithm, based on the
 * provided C/C++ methodology (Tasks 1-6).
 *
 * This class handles:
 * - Tasks 1-4: Initialization, data loading, and termination.
 * - Task 5: Gaussian elimination (forward elimination).
 * - Task 6: Back substitution.
 */
public class SerialGauss {

    // --- Task 1: Variable Declaration ---
    private double[] pMatrix; // Matrix of the linear system (A)
    private double[] pVector; // Right parts of the linear system (b)
    private double[] pResult; // Result vector (x)
    private int size;         // Size of the matrix and vectors

    // --- Task 5: Pivoting Arrays ---
    private int[] pSerialPivotPos; // Row index selected as pivot for iteration i
    private int[] pSerialPivotIter; // Iteration when row i was used as a pivot

    // Utilities for input and random number generation
    private final Scanner scanner = new Scanner(System.in);
    private final Random random = new Random();

    /**
     * Main entry point of the program.
     */
    public static void main(String[] args) {
        System.out.println("Serial Gauss algorithm for solving linear systems \n");

        SerialGauss gaussSolver = new SerialGauss();

        // --- Task 2 & 3: Initialization ---
        gaussSolver.processInitialization();

        // --- Task 3: Matrix and vector output (before elimination) ---

         System.out.println("\nInitial Matrix:");
         gaussSolver.printMatrix();
         System.out.println("\nInitial Vector:");
         gaussSolver.printVector();

        // --- Task 5 & 6: Execution of Gauss algorithm ---
        gaussSolver.serialResultCalculation();

        // --- Task 6: Printing the result vector ---
        System.out.println("\nResult Vector:");
        gaussSolver.printVector(gaussSolver.pResult); // Print the result

        // --- Task 4: Process termination ---
        gaussSolver.processTermination();

        // Pauses the program before exiting, similar to getch()
        System.out.println("\nPress Enter to terminate the program...");
        try {
            System.in.read();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * --- Task 5: Function for the execution of Gauss algorithm ---
     */
    public void serialResultCalculation() {
        // --- Task 5: Memory allocation for pivoting arrays ---
        pSerialPivotPos = new int[size];
        pSerialPivotIter = new int[size];
        for (int i = 0; i < size; i++) {
            pSerialPivotIter[i] = -1; // Initialize to -1 (not used)
        }

        // --- Task 5: Gaussian elimination (Forward) ---
        serialGaussianElimination();

        // --- Task 6: Back substitution ---
        serialBackSubstitution();

        // --- Task 5: Memory deallocation ---
        pSerialPivotPos = null;
        pSerialPivotIter = null;
    }

    /**
     * --- Task 5: Function for the Gaussian elimination ---
     */
    public void serialGaussianElimination() {
        int pivotRow;
        for (int iter = 0; iter < size; iter++) {
            // Finding the pivot row
            pivotRow = findPivotRow(iter);
            pSerialPivotPos[iter] = pivotRow;
            pSerialPivotIter[pivotRow] = iter;

            // Serial column elimination
            serialColumnElimination(iter, pivotRow);
        }

        // This print statement is from Task 5 (Figure 3.6)
        System.out.println("\nMatrix after elimination:");
        printMatrix();
    }

    /**
     * --- Task 5: Function for finding the pivot row ---
     * Finds the row with the max element in the current column (iter).
     */
    public int findPivotRow(int iter) {
        int pivotRow = -1;
        double maxValue = -1.0;

        for (int i = 0; i < size; i++) {
            // Check if row 'i' has not been used as a pivot yet
            if (pSerialPivotIter[i] == -1) {
                double absVal = Math.abs(pMatrix[i * size + iter]);
                if (absVal > maxValue) {
                    pivotRow = i;
                    maxValue = absVal;
                }
            }
        }
        return pivotRow;
    }

    /**
     * --- Task 5: Function for the column elimination ---
     */
    public void serialColumnElimination(int iter, int pivotRow) {
        double pivotValue = pMatrix[pivotRow * size + iter];
        double pivotFactor;

        for (int i = 0; i < size; i++) {
            // If row 'i' has not been used as a pivot
            if (pSerialPivotIter[i] == -1) {
                pivotFactor = pMatrix[i * size + iter] / pivotValue;

                for (int j = iter; j < size; j++) {
                    pMatrix[i * size + j] -= pivotFactor * pMatrix[pivotRow * size + j];
                }

                // Modify the vector element
                pVector[i] -= pivotFactor * pVector[pivotRow];
            }
        }
    }

    /**
     * --- Task 6: Function for the back substitution ---
     */
    public void serialBackSubstitution() {
        int rowIndex;
        for (int i = size - 1; i >= 0; i--) {
            // Get the pivot row for the i-th variable
            rowIndex = pSerialPivotPos[i];

            rowIndex = pSerialPivotPos[i];
            pResult[i] = pVector[rowIndex] / pMatrix[rowIndex * size + i];

            for (int j = 0; j < i; j++) {
                int row = pSerialPivotPos[j];
                pVector[row] -= pMatrix[row * size + i] * pResult[i];
            }
        }
    }

    // --- Utility and Initialization Functions (from Tasks 2-4) ---

    public void processInitialization() {
        do {
            System.out.print("Enter the size of the matrix and the vector: ");
            while (!scanner.hasNextInt()) {
                System.out.println("That's not a valid number. Please enter an integer.");
                System.out.print("Enter the size of the matrix and the vector: ");
                scanner.next();
            }
            size = scanner.nextInt();
            if (size <= 0) {
                System.out.println("Size of objects must be greater than 0!");
            }
        } while (size <= 0);

        System.out.printf("Chosen size = %d\n", size);

        pMatrix = new double[size * size];
        pVector = new double[size];
        pResult = new double[size];


        // dummyDataInitialization();
        randomDataInitialization();
    }

    public void dummyDataInitialization() {
        System.out.println("Initializing with dummy data...");
        for (int i = 0; i < size; i++) {
            pVector[i] = i + 1;
            for (int j = 0; j < size; j++) {
                if (j <= i) {
                    pMatrix[i * size + j] = 1;
                } else {
                    pMatrix[i * size + j] = 0;
                }
            }
        }
    }

    public void randomDataInitialization() {
        System.out.println("Initializing with random data...");
        for (int i = 0; i < size; i++) {
            pVector[i] = random.nextDouble() * 100;
            for (int j = 0; j < size; j++) {
                pMatrix[i * size + j] = random.nextDouble() * 100;
            }
        }
    }

    public void printMatrix() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                System.out.printf("%-10.4f ", pMatrix[i * size + j]);
            }
            System.out.println();
        }
    }

    // Overloaded printVector to print any vector
    public void printVector(double[] vector) {
        if (vector == null) return;
        for (int i = 0; i < size; i++) {
            System.out.printf("%-10.4f\n", vector[i]);
        }
    }

    // Original printVector prints the class member pVector
    public void printVector() {
        printVector(this.pVector);
    }

    public void processTermination() {
        pMatrix = null;
        pVector = null;
        pResult = null;
        System.out.println("\nProcess terminated and resources marked for cleanup.");
    }
}