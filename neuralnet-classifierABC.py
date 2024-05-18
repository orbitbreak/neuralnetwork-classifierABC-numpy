# Numpy implementation of neural network classifier for 3 letters: A/B/C
#   - Labeled dataset of letters A/B/C represented as matrix of 1's and 0's
#   - 3 layer structure:
#       layer 1: inputs layer (1,30)
#       layer 2: hidden layer (1,5)
#       layer 3: output layer (3,3) # classifies input as letter A/B/C (prediction)
#   - Script workflow: labeled_data => generate_weights => train => predict

import numpy as np
import matplotlib.pyplot as plt

##############################################
# Data setup and numpy arrays init
##############################################

# Labeled dataset of letters A/B/C
letter_A = [0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1]
letter_B = [0, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 0]
letter_C = [0, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0]
# Creating labels
letter_labels =[[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]

# print letter A representation
plt.imshow(np.array(letter_A).reshape(5,6))
plt.title("Labeled data of letter_A, close to continue training")
plt.show()

# flattening letter matrices to 1D numpy array
list_of_letter_flatvecs = [ np.array(letter_A).reshape(1,30),
                            np.array(letter_B).reshape(1,30),
                            np.array(letter_C).reshape(1,30)]
# convert letter_labels to numpy array
letter_labels = np.array(letter_labels)
print("\nlist_of_letter_flatvecs:\n", list_of_letter_flatvecs, "\n\n", "letter_labels:\n", letter_labels)

##############################################
# Neural network logic
##############################################

# activation function, use sigmoid S-shape
def my_sigmoid(x):
    return( 1/(1 + np.exp(-x)) )

# feed-forward network:
#   layer 1 input (1,30)
#   layer 2 hidden (1,5)
#   layer 3 output (3,3)
def feed_forward(x, w1, w2):
    #print("...top of feed_foward...")
    # hidden layer 2
    z1 = x.dot(w1)      # input from layer 1
    a1 = my_sigmoid(z1) # output from layer 2

    # output layer 3
    z2 = a1.dot(w2)     # input to layer 3
    a2 = my_sigmoid(z2) # output from layer 3
    return(a2)

# weights init random
def generate_weights(x,y):
    #print("...top of generate_weights...")
    templist = []
    for i in range(x*y):
        templist.append( np.random.rand() )
    return( np.array(templist).reshape(x,y) )

# loss using mean-square-error
def loss_calc(out, y):
    #print("...top of loss_calc...")
    s = np.square(out-y)
    s = np.sum(s)/len(letter_labels)
    return(s)

# error back-propagation
def error_back_propagation(x, y, w1, w2, alpha):
    #print("...top of error_back_propagation...")
    # hidden layer 2
    z1 = x.dot(w1)      # layer 1 input
    a1 = my_sigmoid(z1) # output from layer 2
    # output layer 3
    z2 = a1.dot(w2)     # layer 3 input
    a2 = my_sigmoid(z2) # layer 3 output
    # error in output layer 3
    d2 = (a2 - y)
    d1 = np.multiply( (w2.dot((d2.transpose()))).transpose(),
                       np.multiply(a1, 1 - a1) )
    # gradient for weights w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
    # updating weights
    w1 = w1 - (alpha * (w1_adj))
    w2 = w2 - (alpha * (w2_adj))
    return(w1,w2)

# training algo
def train( x, y, w1, w2, alpha=0.01, epoch=10):
    #print("...top of train...")
    acc = []
    loss = []
    for j in range(epoch):
        templist = []
        for i in range(len(x)):
            out = feed_forward( x[i], w1, w2)
            templist.append( (loss_calc(out, y[i]) ) )
            w1, w2 = error_back_propagation( x[i], letter_labels[i], w1, w2, alpha)
        print("epoch:", j+1, "******** acc:", (1 - (sum(templist)/len(x)) )*100 )
        acc.append( (1-(sum(templist)/len(x)))*100 )
        loss.append( sum(templist)/len(x))
    return(acc, loss, w1, w2)

# prediction
def predict(x, w1, w2):
    #print("...top of predict...")
    output = feed_forward(x, w1, w2)
    maxval=0
    k=0
    for i in range(len(output[0])):
        if (maxval < output[0][i]):
            maxval = output[0][i]
            k = i
    print("\n In predict, found k=", k)
    if (k==0):
        print("Predicting image is a letter A")
    if (k==1):
        print("Predicting image is a letter B")
    if (k==2):
        print("Predicting image is a letter C")
    plt.imshow(x.reshape(5,6))
    plt.show()


##############################################
# Script main
# from labeled_data => generate_weights => train => predict
##############################################

w1 = generate_weights(30, 5) # layer 1 input to layer 2 hidden
w2 = generate_weights(5, 3)  # layer 2 hidden to layer 3 output
print(w1, "\n\n", w2)

# INPUTS:
# dataset list_of_letter_flatvecs as x, true letter_labels as y, weights w1 and w2, learning rate = 0.1, number of iterations = 100
# OUTPUTS:
# accuracy matrix, loss, trained weights w1 and w2
acc, loss, w1, w2 = train(list_of_letter_flatvecs, letter_labels, w1, w2, 0.1, 1000)

# plotting accuracy and loss at each epoch (iteration)
plt.plot(acc)
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.show()

plt.plot(loss)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()

# trained weights matrix
print("\nTrained weights:\nw1:\n", w1, "\nw2:\n", w2)

# classification/prediction
print("\nChecking image of letter B, hopefully it guesses B...")
predict(list_of_letter_flatvecs[1], w1, w2) # B
print("\nChecking image of letter C, hopefully it guesses C...")
predict(list_of_letter_flatvecs[2], w1, w2) # C
print("\nChecking image of letter A, hopefully it guesses A...")
predict(list_of_letter_flatvecs[0], w1, w2) # A


