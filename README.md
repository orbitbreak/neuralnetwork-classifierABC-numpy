Numpy implementation of neural network classifier for 3 letters: A/B/C
  - Labeled dataset of letters A/B/C represented as matrix of 1's and 0's
  - 3 layer structure:
        layer 1: inputs layer (1,30)
        layer 2: hidden layer (1,5)
        layer 3: output layer (3,3) # classifies input as letter A/B/C (prediction)
  - Script workflow: labeled_data => generate_weights => train => predict
