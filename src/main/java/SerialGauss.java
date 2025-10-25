import java.util.Scanner;
import java.util.Random;

public class SerialGauss {
    private double[] pMatrix; // Matrix of the linear system (A)
    private double[] pVector; // Right parts of the linear system (b)
    private double[] pResult; // Result vector (x)
    private int size;         // Size of the matrix and vectors

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

        // --- Task 3: Matrix and vector output ---
        System.out.println("\nInitial Matrix:");
        gaussSolver.printMatrix();
        System.out.println("\nInitial Vector:");
        gaussSolver.printVector();

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
     * --- Task 2 & 3: Function for memory allocation and data initialization ---
     */
    public void processInitialization() {
        // --- Task 2: Setting the size of the matrix and the vector ---
        do {
            System.out.print("Enter the size of the matrix and the vector: ");

            // Basic input validation to ensure it's an integer
            while (!scanner.hasNextInt()) {
                System.out.println("That's not a valid number. Please enter an integer.");
                System.out.print("Enter the size of the matrix and the vector: ");
                scanner.next(); // Discard the invalid input
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


        dummyDataInitialization();

        // randomDataInitialization();
    }

    /**
     * --- Task 3: Function for simple initialization (Dummy Data) ---
     */
    public void dummyDataInitialization() {
        System.out.println("Initializing with dummy data...");
        for (int i = 0; i < size; i++) {
            // pVector[i] = 1, 2, 3, ...
            pVector[i] = i + 1;
            for (int j = 0; j < size; j++) {
                // Accessing the 1D array as a 2D matrix (row-major order)
                if (j <= i) {
                    pMatrix[i * size + j] = 1; // Lower triangular matrix of 1s
                } else {
                    pMatrix[i * size + j] = 0;
                }
            }
        }
    }

    /**
     * --- Task 3 (Alt): Function for random initialization ---
     */
    public void randomDataInitialization() {
        System.out.println("Initializing with random data...");
        for (int i = 0; i < size; i++) {
            // Equivalent to rand() / double(1000) - generates a random double
            pVector[i] = random.nextDouble() * 100; // Using a 0-100 range
            for (int j = 0; j < size; j++) {
                if (j <= i) {
                    pMatrix[i * size + j] = random.nextDouble() * 100;
                } else {
                    pMatrix[i * size + j] = 0;
                }
            }
        }
    }

    /**
     * --- Task 3: Function to print the matrix ---
     */
    public void printMatrix() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Using printf for formatted output, similar to C
                System.out.printf("%-8.2f ", pMatrix[i * size + j]);
            }
            System.out.println();
        }
    }

    /**
     * --- Task 3: Function to print the vector ---
     */
    public void printVector() {
        for (int i = 0; i < size; i++) {
            System.out.printf("%-8.2f\n", pVector[i]);
        }
    }

    /**
     * --- Task 4: Function for process termination ---
     */
    public void processTermination() {
        // Setting references to 'null' can help the GC, but it's not
        // strictly necessary for small programs.
        pMatrix = null;
        pVector = null;
        pResult = null;

        System.out.println("\nProcess terminated and resources marked for cleanup.");
    }
}