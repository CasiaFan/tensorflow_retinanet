from collections import OrderedDict
import numpy as np
import tensorflow as tf

from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner

""" 
See TensorFlow Object Detection API multiscale_grid_anchor_generator.py for detail info 
"""
class Anchor():
    def __init__(self, feature_map_shape_list,
                       img_size,
                       min_level=3,
                       max_level=7,
                       anchor_scale=4.0,
                       aspect_ratios=(1.0, 2.0, 0.5),
                       scales_per_octave=2):
        """
        Arg:
            feature_map_shape_list: list of pairs of conv layer resolutions in format [(h0, w0), (h1, w1), ...]
            img_size: size of image to generate the grid for
            min_level: min level of output feature pyramid
            max_level: max level of output feature pyramid
            anchor_scale: anchor scale and feature stride define the size of base anchor on the image. For example,
                          a feature pyramid with strides [2^3, ..., 2^7] and anchor size 4,
                          the base anchor size is 4 * [2^3, ..., 2^7]
            aspect_ratios: tuple of aspect ratios to place on each grid point
            scales_per_octave: num of intermediate scales per scale octave
        """
        self._anchor_grid_info = []
        self._aspect_ratios = aspect_ratios
        self._scales_per_octave = scales_per_octave
        self._feature_map_shape_list = feature_map_shape_list
        self._im_height, self._im_width = img_size
        scales = [2**(float(scale) / scales_per_octave) for scale in range(scales_per_octave)]
        aspects = list(aspect_ratios)
        for level in range(min_level, max_level+1):
            anchor_stride = [2**level, 2**level]
            base_anchor_size = [2**level*anchor_scale, 2**level*anchor_scale]
            self._anchor_grid_info.append({'level': level, 'info': [scales, aspects, base_anchor_size, anchor_stride]})

    def num_anchors_per_location(self):
        return len(self._aspect_ratios) * self._scales_per_octave

    def _generate(self):
        """
        Generate a list of boundling boxes to be used as anchors
        returns:
            anchor_boxes: list of boxes with shape (N, 4) for [xmin, ymin, xmax, ymax] in normalized coordinates
        """
        anchor_grid_list = []
        for feature_shape, grid_info in zip(self._feature_map_shape_list, self._anchor_grid_info):
            level = grid_info["level"]
            stride = 2**level
            scales, aspects, base_anchor_size, anchor_stride = grid_info["info"]
            anchor_offset = [0, 0]
            if self._im_height % float(stride) == 0 or self._im_height == 1:
                anchor_offset[0] = stride / 2.0
            if self._im_width % float(stride) == 0 or self._im_width == 1:
                anchor_offset[1] = stride / 2.0
            # get each anchor width and height
            scales_grid, aspect_ratio_grid = np.meshgrid(scales, aspects)
            scales_grid = np.reshape(scales_grid, -1)
            aspect_ratio_grid = np.reshape(aspect_ratio_grid, -1)
            as_sqrt = np.sqrt(aspect_ratio_grid)
            anchor_heights = scales_grid / as_sqrt * base_anchor_size[0]
            anchor_widths = scales_grid * as_sqrt * base_anchor_size[1]
            # get a grid of box centers
            feature_map_h, feature_map_w = feature_shape
            x_centers = np.arange(feature_map_w)
            x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
            y_centers = np.arange(feature_map_h)
            y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
            x_centers, y_centers = np.meshgrid(x_centers, y_centers)
            # get anchor width and height
            anchor_widths, x_centers = np.meshgrid(anchor_widths, x_centers)
            anchor_heights, y_centers = np.meshgrid(anchor_heights, y_centers)
            # normalized
            y_centers, anchor_heights = y_centers / self._im_height, anchor_heights / self._im_height
            x_centers, anchor_widths = x_centers / self._im_width, anchor_widths / self._im_width
            box_centers = np.stack([y_centers, x_centers], axis=2)
            box_sizes = np.stack([anchor_heights, anchor_widths], axis=2)
            box_centers = np.reshape(box_centers, [-1, 2])
            box_sizes = np.reshape(box_sizes, [-1, 2])
            box_corners = np.concatenate([box_centers-0.5*box_sizes, box_centers+0.5*box_sizes], axis=1)
            box_corners = tf.cast(box_corners, tf.float32)
            anchor_grid_list.append(box_corners)
        return tf.cast(tf.concat(anchor_grid_list, axis=0), tf.float32)

    @property
    def boxes(self):
        return self._generate()


def create_target_assigner(match_threshold=0.5, unmatched_cls_target=None):
    """
    match_threshold: threshold to assign positive labels for anchors
    unmatched_class_label: tensor with shape same as classification target for each anchor or empty for scalar targets
    """
    similarity_calc_func = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=match_threshold,
                                           unmatched_threshold=match_threshold,
                                           negatives_lower_than_unmatched=True,
                                           force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    return target_assigner.TargetAssigner(similarity_calc=similarity_calc_func,
                                          matcher=matcher,
                                          box_coder=box_coder,
                                          negative_class_weight=1.0,
                                          unmatched_cls_target=unmatched_cls_target)


def batch_assign_targets(target_assigner,
                         anchors,
                         gt_box_batch,
                         gt_cls_batch):
    """
    Batched assignment of classificaiton and regression targets
    Args:
        target_assigner: target assigner object
        anchors: anchor boxlist with shape (num_anchors, 4)
        gt_box_batch: list of gt_box boxlist with length batch_size
        gt_cls_batch: list of gt_cls of tensors with shape (batch_size, N, num_classes+1)
    Returns:
        batch_cls_target: cls tensor with shape (batch_size, num_anchors, num_classes+1)
        batch_reg_target: box regression tensor with shape (batch_size, num_anchors, 4)
        match_list: list of matcher.Match object indicating the math between anchors and gt_boxes
    """
    if not isinstance(anchors, list):
        anchors = len(gt_box_batch) * [anchors]
    cls_target_list = []
    cls_weight_list = []
    box_target_list = []
    box_weight_list = []
    match_list = []
    for anchor, gt_box, gt_cls in zip(anchors, gt_box_batch, gt_cls_batch):
        cls_target, cls_weight, box_target, box_weight, matches = target_assigner.assign(anchor, gt_box, gt_cls)
        cls_target_list.append(cls_target)
        cls_weight_list.append(cls_weight)
        box_target_list.append(box_target)
        box_weight_list.append(box_weight)
        match_list.append(matches)
    return (tf.stack(cls_target_list), tf.stack(cls_weight_list),
            tf.stack(box_target_list), tf.stack(box_weight_list), match_list)
