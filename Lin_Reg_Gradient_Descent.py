#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:16:40 2017

@author: JD
"""

"""
Implement Linear Regression using Gradient Descent to establish
a function that predicts exam scores based on hours studied.
'Possibly an overkill for simple Linear Regression, but it's an
exercise to practice Gradient Descent'
"""
import numpy as np

def compute_current_error(b, m, data):
    error = 0
    for i in range(len(data)):
        x_i = data[i, 0]
        y_i = data[0, i]
        error += np.square((y_i - (m * x_i + b)))
    sse = error / float(len(data))
    return sse

def step_gradient(b_current, m_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data))
    for i in range(len(data)):
        # Compute gradient of m and b
        m_gradient += (data[i,0])*(data[0,i]-(m_current*data[i,0])+b_current)
        b_gradient += (data[0,i]-(m_current*data[i,0])+b_current)
    # Update m and b by the gradient times a pre-det. learning rate value
    new_b = b_current - (learning_rate * (-2/N*b_gradient))
    new_m = m_current - (learning_rate * (-2/N*m_gradient))
    return (new_b, new_m)

def gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(data), learning_rate)
    return (b,m)

def run():
    data = np.genfromtxt('data.csv', delimiter=',')
    # hyperparameters
    learning_rate = 0.0001
    # y = mx +b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_iterations)
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_current_error(initial_b, initial_m, data)))
    print("Running...")
    [b, m] = gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_current_error(b, m, data)))

