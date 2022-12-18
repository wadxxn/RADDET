# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import torch
from torch import nn
import model.layers as L
import util.helper as helper


class RADDetCart(nn.Module):
    def __init__(self, config_model, config_data, config_train, \
                        anchor_boxes, input_shape):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables 
        could be initialized """
        super(RADDetCart, self).__init__()
        assert (isinstance(input_shape, tuple) or isinstance(input_shape, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        

    def forward(self,x):
        """ attention: building the model at last few lines of this 
        function is important """
        
        
        
        ### NOTE: channel-wise MLP ###
        
        x = torch.reshape(x, [-1, int(x.shape[1]), \
                                int(x.shape[2]*x.shape[3])])

        x = L.denseLayer(x, int(x.shape[-1]),int(x.shape[-1]*2))
        x = L.denseLayer(x, int(x.shape[-1]*2),int(x.shape[-1]))
        x = torch.reshape(x, [-1, int(x.shape[1]), \
                    int(self.input_size[0]), int(self.input_size[1]*2)])
        

        ### NOTE: residual block ###
        conv_shortcut=x
        x = L.convolution2D(x, x.shape[1], 3, \
                                (1,1),  use_bias=True, bn=True)
        x = L.convolution2D(x, x.shape[1], 3, \
                                (1,1),  use_bias=True, bn=True)
        x = L.convolution2D(x, x.shape[1], 1, \
                                (1,1),  use_bias=True, bn=True)
        x = x + conv_shortcut

        ### NOTE: yolo head ###
        x = L.convolution2D(x, int(x.shape[1] * 2), \
                3, (1,1),  use_bias=True, bn=True)
        x = L.convolution2D(x, len(self.anchor_boxes) * (self.num_class + 5), \
                1, (1,1), use_bias=True, bn=False)
        x = torch.reshape(x, [-1] + list(x.shape[1:-1]) + \
                            [len(self.anchor_boxes), self.num_class + 5])
                
        return x

    def decodeYolo(self, yolo_raw):
        output_size = [int(self.config_model["input_shape"][0]), \
                        int(2*self.config_model["input_shape"][0])]
        strides = np.array(output_size) / np.array(list(yolo_raw.shape[1:3]))
        raw_xy, raw_wh, raw_conf, raw_prob = torch.split(yolo_raw, \
                                            (2,2,1,self.num_class), dim=-1)

        xy_grid = torch.meshgrid(torch.arange(yolo_raw.shape[1]), torch.arange(yolo_raw.shape[2]))
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, -1), 2)
        xy_grid = torch.transpose(xy_grid, 0,1)
        xy_grid = torch.tile(torch.unsqueeze(xy_grid, 0), \
                        [torch.shape(yolo_raw)[0], 1, 1, len(self.anchor_boxes), 1])
        xy_grid = xy_grid.type(torch.float32)

        scale = self.config_model["yolohead_xyz_scales"][0]
        ### TODO: not sure about this SCALE, but it appears in YOLOv4 tf version ###
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides

        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        ### NOTE: box regression (YOLOv1 Loss Function) ###
        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) + \
                torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))

        ### NOTE: focal loss function ###
        pred_box=torch.unsqueeze(pred_box,4)
        raw_boxes=torch.unsqueeze(raw_boxes,1)
        raw_boxes=torch.unsqueeze(raw_boxes,2)
        raw_boxes=torch.unsqueeze(raw_boxes,3)
        iou = helper.tf_iou2d(pred_box,\
                    raw_boxes)
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1), -1)
        gt_conf_negative = (1.0 - gt_conf) * ((max_iou < self.config_train["focal_loss_iou_threshold"]).type(torch.float32))
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01
        conf_loss = conf_focal * (\
                gt_conf * gt_conf*-torch.log(torch.sigmoid(raw_conf)) + (1-gt_conf)*-torch.log(1-torch.sigmoid(raw_conf)) \
                + \
                alpha * gt_conf_negative * \
                        gt_conf*-torch.log(torch.sigmoid(raw_conf)) + (1-gt_conf)*-torch.log(1-torch.sigmoid(raw_conf)))

        ### NOTE: category loss function ###
        category_loss = gt_conf * \
                gt_category*-torch.log(torch.sigmoid(raw_category)) + (1-gt_category)*-torch.log(1-torch.sigmoid(raw_category))
        
        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1,2,3,4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1,2,3,4]))
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1,2,3,4]))
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all

    
