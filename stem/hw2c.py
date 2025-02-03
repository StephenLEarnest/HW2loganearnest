import numpy as np


# Function to check if the matrix is diagonally dominant
def is_diagonally_dominant(A):
    """
    Checks if the given matrix A is diagonally dominant.
    """
    N = A.shape[0]
    for i in range(N):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        if np.abs(A[i, i]) < row_sum:
            return False
    return True


# Function to make the matrix diagonally dominant by row swaps (if needed)
def make_diagonally_dominant(Aaug):
    """
    Modify the augmented matrix Aaug to make the coefficient matrix diagonally dominant.
    Aaug is a matrix of the form [A | b], where A is the coefficient matrix and b is the right-hand side.
    """
    A = Aaug[:, :-1]
    N = A.shape[0]

    for i in range(N):
        # Check if the current row is diagonally dominant
        if np.abs(A[i, i]) < np.sum(np.abs(A[i])) - np.abs(A[i, i]):
            # Try to find a row with a larger diagonal element
            for j in range(i + 1, N):
                if np.abs(A[j, i]) > np.abs(A[i, i]):
                    # Swap rows i and j
                    A[[i, j], :] = A[[j, i], :]
                    Aaug[[i, j], :] = Aaug[[j, i], :]
                    break
    return Aaug


# Gauss-Seidel method to solve the system of linear equations
def GaussSeidel(Aaug, x, Niter=15):
    """
    Use the Gauss-Seidel method to solve the system of linear equations Ax = b.

    Aaug: Augmented matrix containing the coefficient matrix A and the right-hand side vector b.
          It is a 2D numpy array with shape (N, N+1), where N is the number of equations.
    x: Initial guess vector (1D numpy array of length N).
    Niter: Number of iterations to perform (default is 15).

    Returns: The final estimate of the solution vector x after Niter iterations.
    """
    # Extract the number of equations (N) from the shape of Aaug
    N = Aaug.shape[0]

    # Perform the Gauss-Seidel iterations
    for k in range(Niter):
        for i in range(N):
            # Get the current row, excluding the last column (b)
            row = Aaug[i, :-1]

            # Get the value of b from the augmented matrix
            b = Aaug[i, -1]

            # Compute the sum of the known values in the current row
            sum1 = np.dot(row[:i], x[:i])  # Sum over x1 to x(i-1)
            sum2 = np.dot(row[i + 1:], x[i + 1:])  # Sum over x(i+1) to xN

            # Update the value of x[i]
            x[i] = (b - sum1 - sum2) / Aaug[i, i]

        # Optional: Print the solution after each iteration (for debugging/understanding)
        # print(f"Iteration {k+1}: {x}")

    return x


# Example system of equations to solve
def main():
    # Augmented matrix for the system of linear equations
    # Example: 3 equations with 3 unknowns
    # Aaug = [A | b] where A is the coefficient matrix and b is the right-hand side
    Aaug = np.array([[4, -1, 0, 3],
                     [-1, 4, -1, 1],
                     [0, -1, 4, 2]], dtype=float)

    # Initial guess for the solution
    x_init = np.zeros(Aaug.shape[0])

    # Step 1: Make the matrix diagonally dominant (if necessary)
    Aaug = make_diagonally_dominant(Aaug)

    # Step 2: Call the Gauss-Seidel method to solve the system
    solution = GaussSeidel(Aaug, x_init, Niter=15)

    # Step 3: Print the solution
    print("Solution:", solution)


# Call the main function
if __name__ == "__main__":
    main()