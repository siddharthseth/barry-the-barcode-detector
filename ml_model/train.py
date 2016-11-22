import matplotlib
matplotlib.use('Agg')
import numpy as np 
import os
import caffe
import scipy.io as sio
from scipy import misc

import matplotlib.pyplot as plt

NUM_TRAINING_EXAMPLES = 2
caffe.set_device(0)
caffe.set_mode_gpu()

def load_data():
    training_data = []
    training_labels = []
    for i in range(NUM_TRAINING_EXAMPLES):
        example = misc.imread('barry/data/training/training_example_'+str(i)+'.png')
        example = np.transpose(example, (2, 0, 1))
        # example = (example[:, :, 0] + example[:, :, 1] + example[:, :, 2])/3
        label = misc.imread('barry/data/training/training_label_'+str(i)+'.png')
        label = misc.imresize(label, (30, 40))[:, :, 0]
        training_data.append(example)
        training_labels.append(label)
    # import pdb; pdb.set_trace()
    training_data = np.stack(training_data)
    training_labels = np.stack(training_labels)
    return training_data, training_labels

def train_detector(solver_file):
    training_data, training_labels = load_data()
    solver = caffe.get_solver(solver_file)
    net = solver.net
    batch_size = min(NUM_TRAINING_EXAMPLES, 10)
    training_steps = 1000
    net.blobs['image'].reshape(*np.array((batch_size, training_data[0].shape[0], training_data[0].shape[1], training_data[0].shape[2])))
    net.blobs['mask'].reshape(*np.array((batch_size, training_labels[0].shape[0], training_labels[0].shape[1])))
    training_index = 0
    for i in range(training_steps):
        # net = solver.net
        images = training_data[training_index*batch_size:(training_index+1)*batch_size]
        masks = training_labels[training_index*batch_size:(training_index+1)*batch_size]
        # training_index = (training_index + 1) % training_batches

        net.blobs['image'].data[...] =  images
        net.blobs['mask'].data[...] = masks
        solver.step(1)

train_detector('barry/detector_solver.prototxt')
