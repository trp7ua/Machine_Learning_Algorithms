from random import sample
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

N = 98 # the size of the training set
num_test_vals = 25 # the size of the test set
test_size = 10  # the number of elements to use from the test set
lambdas = [1e-3, 1e-2, 1e-1, 1]
num_epochs = 5
num_steps = 10

def main():
  for i in lambdas:
    train_model(i)

def train_model(lambda_val):
  x = []
  y = []
  u = np.array([1,1,0,1,1,0,1,1,1,0,0,1,1,0]) # random guess of starting parameters
  a = u[0:13]
  b = u[11]
  plot_count = 0
  random_elems = sample(range(N, N + num_test_vals), test_size) # choose a random set of test_size elements from the test values
  for epoch in range(1, num_epochs + 1):
    step_length = 1 / (0.01 * epoch + 50)
    for step in range(1, num_steps + 1):
      with open('./data/binary_train.csv', 'r') as trainingfile:
        trainingreader = reader(trainingfile)
        random_row = sample(range(N), 1) # choose a random element(s) to train the model with
        row_num = 0
        for row in trainingreader:
          if row_num in random_row:
            x_k = np.array(map(float, row[1:])) # convert from str to int array
            y_k = int(row[0])
            threshold = (np.dot(x_k, a) + b) * y_k
            if threshold >= 1:  #calculate gradient
              a = a - (step_length * lambda_val * a)
              b = b
            else:
              gradient = (lambda_val * a) - (y_k * x_k)
              gradient = (step_length * gradient)
              a = a - gradient
              b = b - step_length * -y_k
          row_num = row_num + 1
      if step % 10 is 0:
        print a
        percentage = check_accuracy(a, b, random_elems)
        plot_count = plot_count + 1
        x.append(plot_count)
        y.append(percentage)
  plt.plot(x, y, 'ro')
  plt.show()

def check_accuracy(a, b, random_elems):
  count = 0
  num_correct = 0.0
  with open('./data/binary_test.csv', 'r') as validationfile:
    validationreader = reader(validationfile)
    #row_num = 0
    for row in validationreader:
      #if row_num in random_elems:
      row = np.array(map(float, row[:]))
      model_sign = np.sign(np.dot(row[1:], a) + b) 
      actual_sign = np.sign(int(row[0]))
      if model_sign == actual_sign:
        num_correct = num_correct + 1
      count = count + 1
      #row_num = row_num + 1
  percentage = num_correct / count
  #percentage = num_correct
  return percentage

if __name__ == "__main__":
  main()