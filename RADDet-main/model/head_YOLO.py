# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np

import torch
import model.layers as L
import util.helper as helper


def singleLayerHead(feature_map, num_anchors_layer, num_class, last_channel):
    """ YOLO HEAD for one specific feature stage (after FPN) """
    assert isinstance(num_anchors_layer, int)
    assert isinstance(num_class, int)
    ### NOTE: 7 means [objectness, x, y, z, w, h ,d]
    final_output_channels = int(last_channel * num_anchors_layer * (num_class + 7))
    final_output_reshape = [-1] + list(feature_map.shape[1:-1]) + \
                        [int(last_channel), int(num_anchors_layer) * (num_class + 7)]
    ### NOTE: either use attention or not ###
    conv = feature_map
    ### NOTE: size up the channels is the way how YOLOv4 did it,
    ### other options may also be worth trying ###
    conv = L.convolution2D(conv, feature_map.shape[-1]*2, \
            3, (1,1), "same", "relu", use_bias=True, bn=True, \
            if_regularization=False)

    conv = L.convolution2D(conv, final_output_channels, \
            1, (1,1), "same", None, use_activation=False, use_bias=True, bn=False, \
            if_regularization=False)
    conv = torch.reshape(conv, final_output_reshape)
    return conv


def boxDecoder(yolohead_output, input_size, anchors_layer, num_class, scale=1.):
    """ Decoder output from yolo head to boxes """
    grid_size = yolohead_output.shape[1:4]
    num_anchors_layer = len(anchors_layer)
    grid_strides = np.array(input_size) / np.array(list(grid_size))
    reshape_size = [torch.shape(yolohead_output)[0]] + list(grid_size) + \
                    [num_anchors_layer, 7+num_class]
    reshape_size = tuple(reshape_size)
    pred_raw = torch.reshape(yolohead_output, reshape_size)
    raw_xyz, raw_whd, raw_conf, raw_prob = torch.split(pred_raw, \
                                        (3,3,1,num_class), dim=-1)

    xyz_grid = torch.meshgrid(torch.arange(grid_size[0]), \
                            torch.arange(grid_size[1]), \
                            torch.arange(grid_size[2]))
    xyz_grid = torch.unsqueeze(torch.stack(xyz_grid, -1), 3)
    ### NOTE: swap axes seems necessary, don't know why ###
    xyz_grid = torch.transpose(xyz_grid,0,1)
    xyz_grid = torch.tile(torch.unsqueeze(xyz_grid, 0), \
                    [torch.shape(yolohead_output)[0], 1, 1, 1,  len(anchors_layer), 1])
    xyz_grid = xyz_grid.type(torch.float32) 

    ### NOTE: not sure about this SCALE, but it appears in YOLOv4 tf version ###
    pred_xyz = ((torch.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * \
                grid_strides

    ###---------------- clipping values --------------------###
    raw_whd = torch.clamp(raw_whd, 1e-12, 1e12)
    ###-----------------------------------------------------###
    pred_whd = torch.exp(raw_whd) * anchors_layer
    pred_xyzwhd = torch.cat([pred_xyz, pred_whd], dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)
    return pred_raw, torch.cat([pred_xyzwhd, pred_conf, pred_prob], dim=-1)


def yoloHead(feature, anchors, num_class):
    """ YOLO HEAD main 
    Args:
        feature_stages      ->      feature stages after FPN, [big, mid, small]
        anchor_stages       ->      how many anchors for each stage, 
                                    e.g. [[0,1], [2,3], [4,5]]
        num_class           ->      number of all the classes
    """
    anchor_num = len(anchors)
    yolohead_raw = singleLayerHead(feature, anchor_num, num_class, \
                                    int(feature.shape[1]/4))
    return yolohead_raw
