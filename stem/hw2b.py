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


# Example function to find the root of
def f(x):
    return x ** 2 - 4  # f(x) = x^2 - 4, root at x = 2 or x = -2


def main():
    # First case: x0=1, x1=2, maxiter=5, xtol=1e-4
    root1 = Secant(f, 1, 2, maxiter=5, xtol=1e-4)
    print(f"Root found with x0=1, x1=2, maxiter=5, xtol=1e-4: {root1}")

    # Second case: x0=1, x1=2, maxiter=15, xtol=1e-8
    root2 = Secant(f, 1, 2, maxiter=15, xtol=1e-8)
    print(f"Root found with x0=1, x1=2, maxiter=15, xtol=1e-8: {root2}")

    # Third case: x0=1, x1=2, maxiter=3, xtol=1e-8
    root3 = Secant(f, 1, 2, maxiter=3, xtol=1e-8)
    print(f"Root found with x0=1, x1=2, maxiter=3, xtol=1e-8: {root3}")


# Run the main function
if __name__ == "__main__":
    main()
