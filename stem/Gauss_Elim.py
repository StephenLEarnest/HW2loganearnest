import numpy as np


# Define the function for Simpson's 1/3 rule integration
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


# Define the Probability function
def Probability(PDF, args, c, GT=True):
    # Extract the mean (mu) and standard deviation (sigma) from args
    mu, sigma = args

    # Define the limits of integration
    lower_limit = mu - 5 * sigma
    upper_limit = c if GT else mu + 5 * sigma  # depending on the GT flag, set the upper limit

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


# Example usage with a Gaussian PDF
def gaussian_pdf(x, mu, sigma):
    """ Gaussian PDF function. """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Test the Probability function
if __name__ == "__main__":
    # Define parameters for the Gaussian distribution
    mu = 0  # Mean
    sigma = 1  # Standard deviation

    # Define the upper limit c
    c = 2

    # Use the Probability function for GT=True (probability of x > c)
    probability_gt = Probability(gaussian_pdf, (mu, sigma), c, GT=True)
    print(f"Probability of x > {c}: {probability_gt}")

    # Use the Probability function for GT=False (probability of x < c)
    probability_lt = Probability(gaussian_pdf, (mu, sigma), c, GT=False)
    print(f"Probability of x < {c}: {probability_lt}")
