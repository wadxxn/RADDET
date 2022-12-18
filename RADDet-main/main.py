# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.utils.data import DataLoader




from dataset.data_generator import TrainDataset
import metrics.mAP as mAP

import util.loader as loader
import util.helper as helper
import util.drawer as drawer


config = loader.readConfig()
config_data = config["DATA"]
config_radar = config["RADAR_CONFIGURATION"]
config_model = config["MODEL"]
config_train = config["TRAIN"]

anchor_boxes = loader.readAnchorBoxes() # load anchor boxes with order
num_classes = len(config_data["all_classes"])

### NOTE: using the yolo head shape out from model for data generator ###
    

### NOTE: preparing data ###
data_generator = TrainDataset(config_data, config_train, config_model, \
                                [None,16,16,4,78], anchor_boxes)
train_loader = DataLoader(data_generator,
                          batch_size = 32,
                          shuffle = True)
if __name__ == "__main__":
    for batch_i, (RAD_data, gt_labels, raw_boxes) in enumerate(train_loader):

    