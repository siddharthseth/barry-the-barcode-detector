import matplotlib
matplotlib.use('Agg')
import numpy as np 
import os
import caffe
import scipy.io as sio
from scipy import misc
import datetime
import matplotlib.pyplot as plt

NUM_VALIDATION_EXAMPLES = 60
NUM_TRAINING_EXAMPLES = 300
caffe.set_device(0)
caffe.set_mode_gpu()

def load_data():
    training_data = []
    training_labels = []
    for i in range(NUM_VALIDATION_EXAMPLES):
        # i = i+NUM_TRAINING_EXAMPLES
        example = misc.imread('barry/data/training/training_example_'+str(i)+'.png')
        if example.shape[0] != 480:
            continue
        example = misc.imresize(example, (120, 160, 3))
        example = np.transpose(example, (2, 0, 1))
        # example = (example[:, :, 0] + example[:, :, 1] + example[:, :, 2])/3
        label = misc.imread('barry/data/training/training_label_'+str(i)+'.png')
        label = misc.imresize(label, (30, 40))[:, :, 0]
        label = (label > 0) * 1
        training_data.append(example)
        training_labels.append(label)
    # import pdb; pdb.set_trace()
    training_data = np.stack(training_data)
    training_labels = np.stack(training_labels)
    return training_data, training_labels

def detect_mask(prototxt, caffemodel, batch_num):
    batch_size = min(NUM_VALIDATION_EXAMPLES, 1)
    net = caffe.Net(prototxt, caffemodel, caffe.TRAIN)
    training_data, training_labels = load_data()
    true_mask = training_labels[batch_num]
    images = training_data[batch_num*batch_size:(batch_num+1)*batch_size]
    masks = training_labels[batch_num*batch_size:(batch_num+1)*batch_size]
    net.blobs['image'].data[...] =  images
    net.blobs['mask'].data[...] = masks
    pre = datetime.datetime.now()
    out = net.forward()
    predicted_mask = net.blobs['decode4'].data > 0.5
    predicted_mask = predicted_mask.reshape((30, 40))
    total = datetime.datetime.now() - pre
    import pdb; pdb.set_trace()

detect_mask("barry/edge_detector.prototxt", "barry/detector_adagrad_train_iter_4000.caffemodel", 0)
detect_mask("barry/edge_detector.prototxt", "barry/detector_adagrad_train_iter_4000.caffemodel", 1)
detect_mask("barry/edge_detector.prototxt", "barry/detector_adagrad_train_iter_4000.caffemodel", 2)
detect_mask("barry/edge_detector.prototxt", "barry/detector_adagrad_train_iter_4000.caffemodel", 3)
detect_mask("barry/edge_detector.prototxt", "barry/detector_adagrad_train_iter_4000.caffemodel", 4)