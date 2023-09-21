from numpy import *
from math import *

def compute_error(b, m, points):
    
    totalError = 0
    for pt in points:
        # Get the x and y values
        x = pt[0]
        y = pt[1]
        
        # compute the squared difference of the acutal point 
        # and the current estimated line: y = mx + b
        totalError += pow((y - ((m * x) + b)), 2)
    
    # Return the average error
    return totalError / float(len(points))

def run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations):
    # Starting b and m (y=mx+b)
    b = initial_b
    m = initial_m
    
    # Gradient Descent
    for i in range(num_iterations):
        # Update b and m with more accurate values by performing
        # a step of gradient descent
        b, m = step_gradient(b, m, array(points), learning_rate)
        
    return [b, m]

def step_gradient(current_b, current_m, points, learning_rate):
    
    # Starting points for our gradients
    b_partial_deriv = 0
    m_partial_deriv = 0
    n = len(points)
    
    for pt in points:
        x = pt[0]
        y = pt[1]
        
        # Determine best direction wrt m and b
        # Computing partial derivative of our error function.
        
        b_partial_deriv += -(2/n) * (y-(current_m * x) + current_b)
        m_partial_deriv += -(2/n) * x * (y-(current_m * x) + current_b)

    # Update our current b and m values using our partial derivatives
    new_b = current_b - (learning_rate * b_partial_deriv)
    new_m = current_m - (learning_rate * m_partial_deriv)

    return[new_b, new_m]

def run():
    
    # Step 1: Collect data
    points = genfromtxt('test_inputs/salary_dataset-yrs_exp_vs_salary.csv', delimiter=',')
    
    # Step 2: Define our hyperparameters
    learning_rate = 0.0001      #How fast our model should converge
    num_iterations = 1000       #How many iterations to perform. Too many causes overflow.
    
    #Linear Model: y = mx + b
    initial_b = 0
    initial_m = 0
    
    # Step 3: Train our model
    print('Starting gradient descent at b = {0}, m={1}, error = {2}'.format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    
    [b, m] = run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)

    print('Ending linear estimation: y = {1}x + {2} with an ERROR of: {3}'.format(num_iterations, m, b, compute_error(b, m, points)))


if __name__ == '__main__':
    run()