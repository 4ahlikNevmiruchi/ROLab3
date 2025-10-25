import mpi.*;

public class Main {
    public static void main(String[] args) throws MPIException {
        // Initialize the MPI environment
        MPI.Init(args);

        // Get this process's rank (its unique ID)
        int rank = MPI.COMM_WORLD.getRank();

        // Get the total number of processes in the group
        int size = MPI.COMM_WORLD.getSize();

        System.out.println("Hello from process " + rank + " of " + size);

        // Clean up the MPI environment
        MPI.Finalize();
    }
}