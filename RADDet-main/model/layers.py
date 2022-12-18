# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import torch
import torch.nn as nn
import numpy as np
import os

def denseLayer(input_tensor, input_channel,output_channel):
    
    output =nn.Linear(input_channel,output_channel)
    leakyrelu = nn.LeakyReLU(0.1)
    return leakyrelu(output(input_tensor))

def convolution2D(input_tensor, filters, kernel_size, strides, use_activation=True,use_bias=True, bn=True):
    """ 2D convolutional layer in this research 
    Args:
        input_tensor        ->          input 2D tensor, [None, w, h, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides)
        
    """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    

    ### NOTE: add regularizer to all layers for reducing overfitting ###
    input_channel = input_tensor.shape[1]
    pad = (kernel_size - 1) // 2
    conv_output = nn.Conv2d(
        in_channels=input_channel,out_channels=filters, kernel_size=kernel_size, stride=strides, padding=pad, 
        bias=use_bias)

    if bn: batchnorm = nn.BatchNorm2d(filters)
    
    if use_activation:
        leakyrelu = nn.LeakyReLU(0.1)
        
        
    return leakyrelu(batchnorm(conv_output(input_tensor)))





def maxPooling2D(input_tensor, strides=(2,2),  pool_size=(2,2)):
    """ Max pooling layer in this research (for 2D tensors) """
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    assert all(isinstance(stride, int) for stride in strides)
    assert isinstance(pool_size, tuple)
    assert len(pool_size) == 2
    assert all(isinstance(pool, int) for pool in pool_size)
    

    pool_output = nn.MaxPool2d(
                    kernel_size=pool_size, stride=strides, padding=int((pool_size - 1) // 2)
                    )(input_tensor)
    return pool_output


def convPooling2D(input_tensor, use_bias=True, bn=True):
    """ Try Pooling on the Azimuth dimension and conv2D on range dimension """
    output_tensor = nn.MaxPool2d(
                    kernel_size=(1,2), strides=(1,2)
                    )(input_tensor)
    output_tensor = convolution2D(output_tensor, output_tensor.shape[1], 3, \
                    (2,1),  use_bias=use_bias, bn=bn)
    return output_tensor


























