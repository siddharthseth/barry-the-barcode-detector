# Learned Model

Model prototxt in nets directory.

Placeholder python layer in layers directory.

Solver prototxt in solvers directory.

To run:

1. Build caffe and create a directory named barry in the same level as the caffe directory. Place the model prototxt, the placeholder python layer and the solver prototxt into the barry directory. 
2. Create a directory called data inside barry, divided into training, validation, and testing data directories, with examples and labels in the same directory.
3. Place train.py and validate.py in the same directory level as caffe and barry.
4. Set PYTHONPATH to include the barry directory.
5. Run train.py to train the model. Validate.py loads the trained parameters and does one classification. 