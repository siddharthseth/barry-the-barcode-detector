import caffe
import os, sys, copy
import numpy as np
import scipy.io as sio

class ImageInput(caffe.Layer):

    def setup(self, bottom, top):
        top[0].reshape(1, 3, 120, 160)
        top[1].reshape(1, 30, 40)
        return

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        return

    def backward(self, top, propagate_down, bottom):
        pass