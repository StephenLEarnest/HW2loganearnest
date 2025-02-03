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


# Example of a Gaussian PDF function
def gaussian_pdf(x, mu, sigma):
    """ Gaussian PDF function. """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Define the main function to call Probability and print the results
def main():
    # Find P(x<105 | N(100, 12.5))
    mu_1, sigma_1 = 100, 12.5
    c_1 = 105
    prob_x_lt_105 = Probability(gaussian_pdf, (mu_1, sigma_1), c_1, GT=False)

    # Find P(x>mu+2*sigma | N(100, 3))
    mu_2, sigma_2 = 100, 3
    c_2 = mu_2 + 2 * sigma_2  # c = mu + 2*sigma
    prob_x_gt_mu_plus_2sigma = Probability(gaussian_pdf, (mu_2, sigma_2), c_2, GT=True)

    # Print the results to the console in the requested format
    print(f"P(x<105|N(100,12.5))={prob_x_lt_105:.2f}")
    print(f"P(x>{mu_2 + 2 * sigma_2}|N(100,3))={prob_x_gt_mu_plus_2sigma:.2f}")


# Run the main function
if __name__ == "__main__":
    main()
