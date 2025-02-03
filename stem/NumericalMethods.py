import numpy as np


# Function to perform Simpson's 1/3 rule for numerical integration
def simpson_integration(func, a, b, n=1000):
    """ Approximate the integral of func from a to b using Simpson's 1/3 rule with n intervals. """
    if n % 2 == 1:
        n += 1  # Simpson's rule requires an even number of intervals

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)

    # Apply Simpson's rule
    integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    return integral


# Function to calculate the probability using a given PDF
def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability of x being less than or greater than c using Simpson's rule.

    PDF: The probability density function (Gaussian PDF in this case)
    args: A tuple containing the mean (mu) and standard deviation (sigma)
    c: The upper limit for integration
    GT: If True, return the probability P(x > c); if False, return P(x < c)

    Returns: The probability.
    """
    mu, sigma = args

    # Define the limits of integration
    lower_limit = mu - 5 * sigma
    upper_limit = c if GT else mu + 5 * sigma  # depending on the GT flag

    # Create the function to integrate by passing the PDF, mu, and sigma
    def integrand(x):
        return PDF(x, mu, sigma)

    # Use Simpson's rule to integrate
    probability = simpson_integration(integrand, lower_limit, upper_limit)

    if GT:
        # If GT=True, probability of x > c is the complement of the probability of x < c
        total_probability = 1.0 - probability
    else:
        # If GT=False, the probability of x < c is just the result of the integration
        total_probability = probability

    return total_probability


# Example of a Gaussian PDF function
def gaussian_pdf(x, mu, sigma):
    """ Gaussian PDF function. """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Secant method to find the root of a function
def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of a function fcn(x) in the neighborhood of x0 and x1.

    fcn: Function for which we want to find the root.
    x0, x1: Initial guesses for the root in the neighborhood of the root.
    maxiter: Maximum number of iterations (defaults to 10).
    xtol: Tolerance for convergence (defaults to 1e-5).

    Returns: Final estimate of the root (most recent new x value).
    """

    # Iterate for the Secant Method
    for i in range(maxiter):
        # Calculate the function values at x0 and x1
        f_x0 = fcn(x0)
        f_x1 = fcn(x1)

        # Compute the next estimate using the Secant formula
        if f_x1 - f_x0 == 0:
            print("Division by zero in Secant method; f(x1) - f(x0) is zero.")
            return None

        # Secant update formula
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        # Check for convergence based on the xtol criterion
        if abs(x_new - x1) < xtol:
            return x_new

        # Prepare for the next iteration
        x0, x1 = x1, x_new

    # If the method didn't converge within the maxiter, return the most recent estimate
    print(f"Secant method did not converge within {maxiter} iterations.")
    return x1


# Example function to find its root using the Secant method
def f(x):
    return x ** 2 - 4  # f(x) = x^2 - 4, root at x = 2 or x = -2


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


# Example usage of Gauss-Seidel method
if __name__ == "__main__":
    # Example for the Probability function
    mu_1, sigma_1 = 100, 12.5
    c_1 = 105
    prob_x_lt_105 = Probability(gaussian_pdf, (mu_1, sigma_1), c_1, GT=False)
    print(f"P(x<105|N(100,12.5))={prob_x_lt_105:.2f}")

    mu_2, sigma_2 = 100, 3
    c_2 = mu_2 + 2 * sigma_2  # c = mu + 2*sigma
    prob_x_gt_mu_plus_2sigma = Probability(gaussian_pdf, (mu_2, sigma_2), c_2, GT=True)
    print(f"P(x>{mu_2 + 2 * sigma_2}|N(100,3))={prob_x_gt_mu_plus_2sigma:.2f}")

    # Example usage of Secant method
    root1 = Secant(f, 1, 2, maxiter=5, xtol=1e-4)
    print(f"Root found with x0=1, x1=2, maxiter=5, xtol=1e-4: {root1}")

    root2 = Secant(f, 1, 2, maxiter=15, xtol=1e-8)
    print(f"Root found with x0=1, x1=2, maxiter=15, xtol=1e-8: {root2}")

    root3 = Secant(f, 1, 2, maxiter=3, xtol=1e-8)
    print(f"Root found with x0=1, x1=2, maxiter=3, xtol=1e-8: {root3}")

    # Example usage of Gauss-Seidel method
    Aaug = np.array([[4, -1, 0, 3],
                     [-1, 4, -1, 1],
                     [0, -1, 4, 2]], dtype=float)
    x_init = np.zeros(Aaug.shape[0])
    solution = GaussSeidel(Aaug, x_init, Niter=15)
    print("Solution:", solution)
