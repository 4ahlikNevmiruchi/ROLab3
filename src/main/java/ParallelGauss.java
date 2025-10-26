import mpi.*;
import java.util.Scanner;
import java.util.Arrays;
import java.util.Random;

/**
 * A Java/MPI implementation of the parallel Gauss algorithm, based on the
 * provided C/MPI methodology.
 * This class holds the state for a single MPI process.
 */
public class ParallelGauss {

    // --- ANSI Color Codes ---
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";

    // --- MPI Process Info ---
    private final int procRank; // Rank of the current process
    private final int procNum;  // Total number of processes

    private boolean showDebugOutput = false;
    private boolean performTest = false;

    // --- Problem Size ---
    private int size;     // Total size of the matrix (N)
    private int rowNum;   // Number of rows this process is responsible for
    private int successfulIterations = 0;

    // --- Data Arrays (Global) ---
    // These are only allocated on procRank 0
    private double[] pMatrix; // The complete matrix A
    private double[] pVector; // The complete vector b
    private double[] pResult; // The complete (permuted) result vector x

    // --- Data Arrays (Local) ---
    // These are allocated on ALL processes
    private double[] pProcRows;    // This process's stripe of matrix A
    private double[] pProcVector;  // This process's stripe of vector b
    private double[] pProcResult;  // This process's stripe of result vector x

    // --- Pivoting and Distribution Arrays ---
    // These are allocated on ALL processes
    private int[] pParallelPivotPos;  // Stores the GLOBAL row index for pivot 'i'
    private int[] pParallelPivotIter; // Stores the iteration 'i' when a LOCAL row was used
    private int[] pProcInd; // Global row index where each process's stripe STARTS
    private int[] pProcNum; // Number of rows for each process
    private double[] pPivotRow; // Buffer to hold the pivot row for broadcast

    // --- Utilities ---
    private final Random random = new Random();

    /**
     * Constructor: Initializes the MPI rank and size.
     */
    public ParallelGauss(int procRank, int procNum) {
        this.procRank = procRank;
        this.procNum = procNum;
    }

    /**
     * Main entry point for the MPI program.
     */
    public static void main(String[] args) throws Exception {
        MPI.Init(args);
        int rank = MPI.COMM_WORLD.getRank();
        int size = MPI.COMM_WORLD.getSize();

        ParallelGauss solver = new ParallelGauss(rank, size);
        solver.run();

        MPI.Finalize();
    }

    /**
     * Executes the main logic of the parallel algorithm.
     * (Corresponds to the C 'main' function).
     */
    public void run() throws MPIException {
        if (procRank == 0) {
            System.out.println("Parallel Gauss algorithm for solving linear systems\n");
        }

        // 1. Initialize memory, get size, and allocate arrays
        processInitialization();

        // 2. Start timer and distribute data
        double startTime = MPI.wtime();
        dataDistribution();
        testDistribution();

        // 3. Perform the parallel Gaussian elimination and back-substitution
        parallelResultCalculation();
        printEliminationResult();

        // 4. Gather the results back to the master process
        resultCollection();

        // 5. Stop timer
        double duration = MPI.wtime() - startTime;

        // 6. Print results and test
        if (procRank == 0) {
            printResultVector();

            if (performTest) {
                testResult();
            }
            else System.out.println(ANSI_RED + "Test has been skipped" + ANSI_RESET + "\n");
            System.out.printf("\nTime of execution: %f\n", duration);
        }

        // 7. Clean up
        processTermination();
    }

    /**
     * C: ProcessInitialization
     * Gets matrix size, validates it, and allocates all arrays.
     */
    public void processInitialization() throws MPIException {
        int[] sizeBuf = new int[1]; // Buffer for broadcasting the size
        int[] testChoiceBuf = new int[1]; // Buffer for broadcasting the user's choice

        if (procRank == 0) {
            Scanner scanner = new Scanner(System.in);
            do {
                System.out.print("Enter the size of the matrix and the vector: \n");
                System.out.flush();
                if (scanner.hasNextInt()) {
                    sizeBuf[0] = scanner.nextInt();
                } else {
                    scanner.next(); // Clear invalid input
                    sizeBuf[0] = 0;
                }
                if (sizeBuf[0] < procNum) {
                    System.out.printf("Size must be greater than number of processes! (%d)\n", procNum);
                }
            } while (sizeBuf[0] < procNum);
            do {
                System.err.print("Perform correctness test? (y/n): \n");
                String input = scanner.next().trim().toLowerCase();
                if (input.equals("y")) {
                    testChoiceBuf[0] = 1; // true
                    break;
                } else if (input.equals("n")) {
                    testChoiceBuf[0] = 0; // false
                    break;
                }
            } while (true);
        }

        // Broadcast the validated size from rank 0 to all other processes
        MPI.COMM_WORLD.bcast(sizeBuf, 1, MPI.INT, 0);
        this.size = sizeBuf[0];

        MPI.COMM_WORLD.bcast(testChoiceBuf, 1, MPI.INT, 0);
        this.performTest = (testChoiceBuf[0] == 1);

        if (this.size <= 10) {
            this.showDebugOutput = true;
        }

        /*

        // --- Calculate row distribution (per C code in ProcessInitialization) ---
        // This is a balanced distribution
        pProcNum = new int[procNum];
        pProcInd = new int[procNum];
        int restRows = size;
        int pProcIndOffset = 0;

        for (int i = 0; i < procNum; i++) {
            pProcNum[i] = restRows / (procNum - i);
            restRows -= pProcNum[i];

            pProcInd[i] = pProcIndOffset;
            pProcIndOffset += pProcNum[i];
        }

        this.rowNum = pProcNum[procRank]; // Local row count

         */

        // --- Calculate row distribution (Simple Block Distribution) ---
        // This is the standard, simple way to distribute rows.
        pProcNum = new int[procNum]; // Number of rows for each process
        pProcInd = new int[procNum]; // Starting row index for each process

        int rowNumSimple = size / procNum;
        int restRows = size % procNum;
        int pProcIndOffset = 0;

        for (int i = 0; i < procNum; i++) {
            pProcNum[i] = (i < restRows) ? (rowNumSimple + 1) : rowNumSimple;
            pProcInd[i] = pProcIndOffset;
            pProcIndOffset += pProcNum[i];
        }

        this.rowNum = pProcNum[procRank]; // This process's number of rows

        // --- Allocate arrays for ALL processes ---
        pProcRows = new double[rowNum * size];
        pProcVector = new double[rowNum];
        pProcResult = new double[rowNum];

        pParallelPivotPos = new int[size]; // Stores global pivot row index
        pParallelPivotIter = new int[rowNum]; // Stores iter when local row was used
        Arrays.fill(pParallelPivotIter, -1);

        pPivotRow = new double[size + 1]; // Buffer for pivot row + vector element

        // --- Allocate arrays for Master Process (Rank 0) only ---
        if (procRank == 0) {
            pMatrix = new double[size * size];
            pVector = new double[size];
            pResult = new double[size]; // This will be the *permuted* result

            // Initialize the matrix and vector with data
            randomDataInitialization(); // Using random data
        }
    }

    /**
     * C: DataDistribution
     * Scatters the matrix and vector from Rank 0 to all processes.
     * NOTE: This function contains its *own* row distribution logic,
     * which (per the C code) overwrites the balanced one from Init
     * for the scatter operation. I am replicating this behavior.
     */
    public void dataDistribution() throws MPIException {

        // Ensure send buffers are non-null on non-root ranks
        // so the native MPI Java binding won't crash.
        if (procRank != 0) {
            // These arrays are ignored by MPI on non-root ranks,
            // but the JNI binding must not receive null.
            pMatrix = new double[0];
            pVector = new double[0];
        }

        // --- 1. Calculate distribution for MATRIX (in elements) ---
        // We use pProcNum/pProcInd (calculated in Init) which are row-based,
        // and convert them to element-based counts/displacements for the matrix.
        int[] pSendNum = new int[procNum];
        int[] pSendInd = new int[procNum];

        for (int i = 0; i < procNum; i++) {
            // THIS IS THE FIX: pProcNum[i] (rows) * size (columns)
            pSendNum[i] = pProcNum[i] * size;
            pSendInd[i] = pProcInd[i] * size;
        }

        // Scatter the matrix rows (pMatrix) to all processes (pProcRows)
        MPI.COMM_WORLD.scatterv(
                pMatrix,    // sendbuf
                pSendNum,   // sendcounts (in elements)
                pSendInd,   // displacements (in elements)
                MPI.DOUBLE, // sendtype
                pProcRows,  // recvbuf
                pSendNum[procRank], // recvcount (this process's element count)
                MPI.DOUBLE, // recvtype
                0           // root
        );

        // --- 2. Scatter the VECTOR (in rows/elements) ---
        // This was already correct, as pProcNum/pProcInd are row-based counts.
        MPI.COMM_WORLD.scatterv(
                pVector,     // sendbuf
                pProcNum,    // sendcounts (number of rows per proc)
                pProcInd,    // displacements (row index offset)
                MPI.DOUBLE,  // sendtype
                pProcVector, // recvbuf
                pProcNum[procRank], // recvcount (this process's row count)
                MPI.DOUBLE,  // recvtype
                0            // root
        );
    }

    /**
     * C: ParallelResultCalculation
     */
    public void parallelResultCalculation() throws MPIException {
        parallelGaussianElimination();
        parallelBackSubstitution();
    }

    /**
     * C: ParallelGaussianElimination
     */
    public void parallelGaussianElimination() throws MPIException {
        int iter;
        for (iter = 0; iter < size; iter++) {
            // --- 1. Find local pivot ---
            double localMaxValue = -1.0;
            int localPivotPos = -1;

            for (int j = 0; j < rowNum; j++) {
                if (pParallelPivotIter[j] == -1) {
                    double absVal = Math.abs(pProcRows[j * size + iter]);
                    if (absVal > localMaxValue) {
                        localMaxValue = absVal;
                        localPivotPos = j;
                    }
                }
            }

            // --- 2. Find global pivot (rank and value) ---
            // This is the Java equivalent of MPI_MAXLOC for (double, int)

            // 2a. Find the global max value
            double[] localMaxBuf = {localMaxValue};
            double[] globalMaxBuf = new double[1];
            MPI.COMM_WORLD.allReduce(localMaxBuf, globalMaxBuf, 1, MPI.DOUBLE, MPI.MAX);
            double globalMaxValue = globalMaxBuf[0];

            if (globalMaxValue < 0.0) {
                if (procRank == 0) {
                    System.err.println(ANSI_RED + "No valid pivot found (matrix may be singular); stopping at iter = " + iter + ANSI_RESET);
                }
                break; // Exit the elimination loop
            }

            // 2b. Find the *lowest* rank that has that max value
            int[] localRankBuf = new int[1];
            int[] globalRankBuf = new int[1];

            localRankBuf[0] = (localMaxValue == globalMaxValue) ? procRank : procNum + 1;
            MPI.COMM_WORLD.allReduce(localRankBuf, globalRankBuf, 1, MPI.INT, MPI.MIN);
            int pivotProcRank = globalRankBuf[0];

            // --- 3. Broadcast pivot info ---
            int[] pivotPosBuf = new int[1]; // Buffer for local pivot position
            if (procRank == pivotProcRank) {
                pivotPosBuf[0] = localPivotPos;
            }
            MPI.COMM_WORLD.bcast(pivotPosBuf, 1, MPI.INT, pivotProcRank);
            int globalPivotPos = pivotPosBuf[0]; // This is the *local* index on the pivotProc

            // --- 4. Store pivot info (all processes do this) ---
            // Store the GLOBAL row index of the pivot
            pParallelPivotPos[iter] = pProcInd[pivotProcRank] + globalPivotPos;

            // Mark the local row as used *only on the process that owns it*
            if (procRank == pivotProcRank) {
                pParallelPivotIter[globalPivotPos] = iter;
            }

            // --- 5. Broadcast pivot row + vector element ---
            if (procRank == pivotProcRank) {
                // Copy matrix row
                System.arraycopy(pProcRows, globalPivotPos * size, pPivotRow, 0, size);
                // Copy vector element
                pPivotRow[size] = pProcVector[globalPivotPos];
            }
            MPI.COMM_WORLD.bcast(pPivotRow, size + 1, MPI.DOUBLE, pivotProcRank);

            // --- 6. Parallel column elimination ---
            parallelEliminateColumns(iter);
        }
    }

    /**
     * C: ParallelEliminateColumns
     */
    public void parallelEliminateColumns(int iter) {
        double multiplier;
        for (int i = 0; i < rowNum; i++) {
            if (pParallelPivotIter[i] == -1) { // If this row is not a pivot row
                multiplier = pProcRows[i * size + iter] / pPivotRow[iter];
                for (int j = iter; j < size; j++) {
                    pProcRows[i * size + j] -= pPivotRow[j] * multiplier;
                }
                pProcVector[i] -= pPivotRow[size] * multiplier;
            }
        }
    }

    /**
     * C: ParallelBackSubstitution
     */
    public void parallelBackSubstitution() throws MPIException {
        double[] iterResultBuf = new double[1]; // Buffer for broadcasting x[i]

        for (int i = size - 1; i >= 0; i--) {
            // --- 1. Find which process holds the pivot row for variable 'i' ---
            int[] pivotInfo = findBackPivotRow(pParallelPivotPos[i]);
            int iterProcRank = pivotInfo[0]; // Rank of process
            int iterPivotPos = pivotInfo[1]; // Local row index on that process

            // --- 2. The owner process calculates the unknown ---
            if (procRank == iterProcRank) {
                double iterResult = pProcVector[iterPivotPos] / pProcRows[iterPivotPos * size + i];
                pProcResult[iterPivotPos] = iterResult;
                iterResultBuf[0] = iterResult;
            }

            // --- 3. Broadcast the calculated unknown (x[i]) to all ---
            MPI.COMM_WORLD.bcast(iterResultBuf, 1, MPI.DOUBLE, iterProcRank);
            double iterResult = iterResultBuf[0];

            // --- 4. Update the vector 'b' on all other rows ---
            for (int j = 0; j < rowNum; j++) {
                // If local row 'j' was a pivot for an earlier iter (k < i)
                if (pParallelPivotIter[j] != -1 && pParallelPivotIter[j] < i) {
                    double val = pProcRows[j * size + i] * iterResult;
                    pProcVector[j] -= val;
                }
            }
        }
    }

    /**
     * C: FindBackPivotRow
     * Helper to find which process rank and local row index
     * correspond to a global row index.
     *
     * @return int[2] where [0] = rank, [1] = local row index
     */
    public int[] findBackPivotRow(int globalRowIndex) {
        int[] result = new int[2];
        for (int i = 0; i < procNum - 1; i++) {
            if (globalRowIndex >= pProcInd[i] && globalRowIndex < pProcInd[i + 1]) {
                result[0] = i; // Process rank
                break;
            }
        }
        if (result[0] == 0 && globalRowIndex >= pProcInd[procNum - 1]) {
            // Edge case: it's in the last process
            result[0] = procNum - 1;
        }

        result[1] = globalRowIndex - pProcInd[result[0]]; // Local row index
        return result;
    }

    /**
     * C: ResultCollection
     * Gathers the permuted result stripes (pProcResult) into
     * the full permuted result vector (pResult) on Rank 0.
     */
    public void resultCollection() throws MPIException {
        // Ensure the receive buffer is non-null on non-root ranks
        // so the native MPI Java binding won't crash.
        if (procRank != 0) {
            // This array is ignored by MPI, but must not be null.
            pResult = new double[0];
        }
        MPI.COMM_WORLD.gatherv(
                pProcResult,   // sendbuf
                rowNum,        // sendcount
                MPI.DOUBLE,    // sendtype
                pResult,       // recvbuf
                pProcNum,      // recvcounts (array)
                pProcInd,      // displacements (array)
                MPI.DOUBLE,    // recvtype
                0              // root
        );
    }

    /**
     * C: TestResult
     * Runs on Rank 0 only. Checks if A * x = b.
     */
    public void testResult() {
        if (procRank != 0) return;

        double[] pRightPartVector = new double[size];
        Arrays.fill(pRightPartVector, 0.0);
        boolean equal = true;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Un-scramble the result vector 'x' using the pivot map
                // pResult[pParallelPivotPos[j]] == x[j]
                pRightPartVector[i] += pMatrix[i * size + j] * pResult[pParallelPivotPos[j]];
            }
            double ACCURACY = 1.0e-6;
            if (Math.abs(pRightPartVector[i] - pVector[i]) > ACCURACY) {
                equal = false;
            }
        }

        if (equal) {
            System.out.println("The result of the parallel Gauss algorithm is " +
                    ANSI_GREEN + "correct." + ANSI_RESET);
        } else {
            System.out.println("The result of the parallel Gauss algorithm is " +
                    ANSI_RED + "NOT correct." + ANSI_RESET + " Check your code.");
        }
    }

    /**
     * C: ProcessTermination
     * In Java, we just nullify references to help the Garbage Collector.
     */
    public void processTermination() {
        if (procRank == 0) {
            pMatrix = null;
            pVector = null;
            pResult = null;
        }
        pProcRows = null;
        pProcVector = null;
        pProcResult = null;
        pParallelPivotPos = null;
        pParallelPivotIter = null;
        pProcInd = null;
        pProcNum = null;
        pPivotRow = null;
    }

    // --- Data Initialization (Serial, Rank 0 only) ---

    /*
    public void dummyDataInitialization() {
        if (procRank != 0) return;
        System.out.println("Initializing with dummy data...");
        for (int i = 0; i < size; i++) {
            pVector[i] = i + 1;
            for (int j = 0; j < size; j++) {
                pMatrix[i * size + j] = (j <= i) ? 1 : 0;
            }
        }
    }
    */

    public void randomDataInitialization() {
        if (procRank != 0) return;

        if (showDebugOutput) {
            System.out.println("Initializing with random data...");
        }

        for (int i = 0; i < size; i++) {
            pVector[i] = random.nextDouble() * 100;
            for (int j = 0; j < size; j++) {
                pMatrix[i * size + j] = random.nextDouble() * 100;
            }
        }
    }

    // --- Printing Functions (Serial, Rank 0 only) ---

    /**
     * C: PrintResultVector (Corrected from main)
     * Prints the final, un-scrambled result vector x.
     */
    public void printResultVector() {
        if (procRank != 0) return;

        if (showDebugOutput) {
            System.out.println("\nResult Vector (x):");
            for (int i = 0; i < size; i++) {
                // Use the pivot map to print x[0], x[1], x[2], ...
                System.out.printf("%10.4f ", pResult[pParallelPivotPos[i]]);
            }
            System.out.println();
        }
    }

    /**
     * C: PrintMatrix
     * (We need this for the debug output)
     */
    public void printMatrix(double[] matrix, int rows, int cols) {
        if (matrix == null) return;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%10.4f ", matrix[i * cols + j]);
            }
            System.out.println();
        }
    }

    /**
     * C: PrintVector
     * (We need this for the debug output)
     */
    public void printVector(double[] vector, int size) {
        if (vector == null) return;
        for (int i = 0; i < size; i++) {
            System.out.printf("%10.4f\n", vector[i]);
        }
    }

    /**
     * C: TestDistribution
     * Prints the initial matrix and the distributed stripes.
     */
    public void testDistribution() throws MPIException {
        if (showDebugOutput) {
            if (procRank == 0) {
                System.out.println("\n--- Initial Matrix (Global) ---");
                printMatrix(pMatrix, size, size);
                System.out.println("\n--- Initial Vector (Global) ---");
                printVector(pVector, size);
            }

            // Use barriers to print each process's stripe in order
            MPI.COMM_WORLD.barrier();
            for (int i = 0; i < procNum; i++) {
                if (procRank == i) {
                    System.out.printf("\n--- Process Rank = %d ---\n", procRank);
                    System.out.println("Matrix Stripe (Local):");
                    printMatrix(pProcRows, rowNum, size);
                    System.out.println("Vector Stripe (Local):");
                    printVector(pProcVector, rowNum);
                }
                MPI.COMM_WORLD.barrier();
            }
        }
    }

    /**
     * Prints the local matrix stripes after elimination.
     */
    public void printEliminationResult() throws MPIException {
        if (showDebugOutput) {
            MPI.COMM_WORLD.barrier(); // Wait for elimination to finish
            for (int i = 0; i < procNum; i++) {
                if (procRank == i) {
                    System.out.printf("\n--- Matrix After Elimination (Rank %d) ---\n", procRank);
                    printMatrix(pProcRows, rowNum, size);
                }
                MPI.COMM_WORLD.barrier(); // Print in order
            }
        }
    }
}