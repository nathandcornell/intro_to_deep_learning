#! python3

from numpy import *

# Computes the error/loss (sum of squared errors)
# This is the sum of the squared distances of each point from the line 
# that our slope indicates.
def compute_error_for_given_points(b, m, points):
    total_error = 0.0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points)) # number of points
    
    # get more optimal b/m for each step
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # Get the partial derivitives
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (rate * b_gradient)
    new_m = m_current - (rate * m_gradient)

    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, rate, iterations):
    b = starting_b
    m = starting_m

    for i in range(iterations):
        b, m = step_gradient(b, m, array(points), rate)
        print(str(compute_error_for_given_points(b, m, array(points))) + "\n")
        
    return [b,m]


def run():
    points = genfromtxt('data.csv', delimiter=',')
    # hyperparameter (how fast it learns - can't be too low or too high)
    # High enough to converge, but not so high that it never completes
    learning_rate = 0.0001

    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0

    # iteration should change as dataset grows/shrinks
    num_iterations = 1000

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)


if __name__ == '__main__':
    run()


