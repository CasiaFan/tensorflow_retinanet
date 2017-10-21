from anchor.anchor_generator import create_retinanet_anchors
from anchor.box_coder import FasterRCNNBoxCoder

import tensorflow as tf

slim = tf.contrib.slim

def AnchorAssign():
    """
    Assign generated anchors to boxes and labels
    """
    def __init__(self, input_size, boxes, labels):
        """
        input_size: input image size
        boxes: boxes coordinates [ymin, ] of given input image
        """