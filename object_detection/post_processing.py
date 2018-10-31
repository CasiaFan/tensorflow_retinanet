# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Post-processing operations on detected boxes."""

import tensorflow as tf

from object_detection import box_list
from object_detection import box_list_ops
from object_detection import shape_utils


def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   scope=None):
  """Multi-class version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates independently for each class for
  which scores are provided (via the scores field of the input box_list),
  pruning boxes with score less than a provided threshold prior to
  applying NMS.

  Please note that this operation is performed on *all* classes, therefore any
  background classes should be removed prior to calling this function.

  Selected boxes are guaranteed to be sorted in decreasing order by score (but
  the sort is not guaranteed to be stable).

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections. The scores have to be non-negative when
      pad_to_max_output_size is True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.
    scope: name scope.

  Returns:
    A tuple of sorted_boxes and num_valid_nms_boxes. The sorted_boxes is a
      BoxList holds M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box. The
      num_valid_nms_boxes is a 0-D integer tensor representing the number of
      valid elements in `BoxList`, with the valid elements appearing first.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if scores.shape.ndims != 2:
    raise ValueError('scores field must be of rank 2')
  if scores.shape[1].value is None:
    raise ValueError('scores must have statically defined second '
                     'dimension')
  if boxes.shape.ndims != 3:
    raise ValueError('boxes must be of rank 3.')
  if not (boxes.shape[1].value == scores.shape[1].value or
          boxes.shape[1].value == 1):
    raise ValueError('second dimension of boxes must be either 1 or equal '
                     'to the second dimension of scores')
  if boxes.shape[2].value != 4:
    raise ValueError('last dimension of boxes must be of size 4.')

  with tf.name_scope(scope, 'MultiClassNonMaxSuppression'):
    num_scores = tf.shape(scores)[0]
    num_classes = scores.get_shape()[1]

    selected_boxes_list = []
    num_valid_nms_boxes_cumulative = tf.constant(0)
    per_class_boxes_list = tf.unstack(boxes, axis=1)
    boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1
                 else [0] * num_classes.value)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
      per_class_boxes = per_class_boxes_list[boxes_idx]
      boxlist_and_class_scores = box_list.BoxList(per_class_boxes)
      class_scores = tf.reshape(
          tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])

      boxlist_and_class_scores.add_field("scores",
                                         class_scores)
      max_selection_size = tf.minimum(max_size_per_class,
                                        boxlist_and_class_scores.num_boxes())
      selected_indices = tf.image.non_max_suppression(
            boxlist_and_class_scores.get(),
            boxlist_and_class_scores.get_field("scores"),
            max_selection_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh)
      num_valid_nms_boxes = tf.shape(selected_indices)[0]
      selected_indices = tf.concat(
            [selected_indices,
             tf.zeros(max_selection_size-num_valid_nms_boxes, tf.int32)], 0)
      nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                       selected_indices)
      # Make the scores -1 for invalid boxes.
      valid_nms_boxes_indx = tf.less(
          tf.range(max_selection_size), num_valid_nms_boxes)
      nms_scores = nms_result.get_field("scores")
      nms_result.add_field("scores",
                           tf.where(valid_nms_boxes_indx,
                                    nms_scores, -1*tf.ones(max_selection_size)))
      num_valid_nms_boxes_cumulative += num_valid_nms_boxes

      nms_result.add_field(
          "classes", (tf.zeros_like(
              nms_result.get_field("scores")) + class_idx))
      selected_boxes_list.append(nms_result)
    selected_boxes = box_list_ops.concatenate(selected_boxes_list)
    sorted_boxes = box_list_ops.sort_by_field(selected_boxes,
                                              "scores")
    if clip_window is not None:
      # When pad_to_max_output_size is False, it prunes the boxes with zero
      # area.
      sorted_boxes = box_list_ops.clip_to_window(
          sorted_boxes,
          clip_window,
          filter_nonoverlapping=True)
      # Set the scores of boxes with zero area to -1 to keep the default
      # behaviour of pruning out zero area boxes.
      sorted_boxes_size = tf.shape(sorted_boxes.get())[0]
      non_zero_box_area = tf.cast(box_list_ops.area(sorted_boxes), tf.bool)
      sorted_boxes_scores = tf.where(
          non_zero_box_area,
          sorted_boxes.get_field("scores"),
          -1*tf.ones(sorted_boxes_size))
      sorted_boxes.add_field("scores", sorted_boxes_scores)
      num_valid_nms_boxes_cumulative = tf.reduce_sum(
          tf.cast(tf.greater_equal(sorted_boxes_scores, 0), tf.int32))
      sorted_boxes = box_list_ops.sort_by_field(sorted_boxes, "scores")

    if max_total_size:
      max_total_size = tf.minimum(max_total_size,
                                  sorted_boxes.num_boxes())
      sorted_boxes = box_list_ops.gather(sorted_boxes,
                                         tf.range(max_total_size))
      num_valid_nms_boxes_cumulative = tf.where(
          max_total_size > num_valid_nms_boxes_cumulative,
          num_valid_nms_boxes_cumulative, max_total_size)

    return sorted_boxes, num_valid_nms_boxes_cumulative


def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         num_valid_boxes=None,
                                         scope=None,
                                         use_static_shapes=False,
                                         parallel_iterations=32):
  """Multi-class version of non maximum suppression that operates on a batch.

  This op is similar to `multiclass_non_max_suppression` but operates on a batch
  of boxes and scores. See documentation for `multiclass_non_max_suppression`
  for details.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
        otherwise, if `q` is equal to number of classes, class-specific boxes
        are used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections. The scores have to be
      non-negative when use_static_shapes is set True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of shape [batch_size, 4]  where each entry is
      of the form [y_min, x_min, y_max, x_max] representing the window to clip
      boxes to before performing non-max suppression. This argument can also be
      a tensor of shape [4] in which case, the same clip window is applied to
      all images in the batch. If clip_widow is None, all boxes are used to
      perform non-max suppression.
    num_valid_boxes: (optional) a Tensor of type `int32`. A 1-D tensor of shape
      [batch_size] representing the number of valid boxes to be considered
      for each image in the batch.  This parameter allows for ignoring zero
      paddings.
    use_static_shapes: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class` and it doesn't clip boxes to max_total_size.
      Defaults to false.
    parallel_iterations: (optional) number of batch items to process in
      parallel.

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'num_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.

  Raises:
    ValueError: if `q` in boxes.shape is not 1 or not equal to number of
      classes as inferred from scores.shape.
  """
  q = boxes.shape[2].value
  num_classes = scores.shape[2].value
  if q != 1 and q != num_classes:
    raise ValueError('third dimension of boxes must be either 1 or equal '
                     'to the third dimension of scores')
  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    boxes_shape = boxes.shape
    batch_size = boxes_shape[0].value
    num_anchors = boxes_shape[1].value

    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    if num_anchors is None:
      num_anchors = tf.shape(boxes)[1]

    # If num valid boxes aren't provided, create one and mark all boxes as
    # valid.
    if num_valid_boxes is None:
      num_valid_boxes = tf.ones([batch_size], dtype=tf.int32) * num_anchors

    if clip_window is None:
      clip_window = tf.stack([
          tf.reduce_min(boxes[:, :, :, 0]),
          tf.reduce_min(boxes[:, :, :, 1]),
          tf.reduce_max(boxes[:, :, :, 2]),
          tf.reduce_max(boxes[:, :, :, 3])
      ])
    if clip_window.shape.ndims == 1:
      clip_window = tf.tile(tf.expand_dims(clip_window, 0), [batch_size, 1])

    def _single_image_nms_fn(args):
      """Runs NMS on a single image and returns padded output.

      Args:
        args: A list of tensors consisting of the following:
          per_image_boxes - A [num_anchors, q, 4] float32 tensor containing
            detections. If `q` is 1 then same boxes are used for all classes
            otherwise, if `q` is equal to number of classes, class-specific
            boxes are used.
          per_image_scores - A [num_anchors, num_classes] float32 tensor
            containing the scores for each of the `num_anchors` detections.
          per_image_clip_window - A 1D float32 tensor of the form
            [ymin, xmin, ymax, xmax] representing the window to clip the boxes
            to.
          per_image_additional_fields - (optional) A variable number of float32
            tensors each with size [num_anchors, ...].
          per_image_num_valid_boxes - A tensor of type `int32`. A 1-D tensor of
            shape [batch_size] representing the number of valid boxes to be
            considered for each image in the batch.  This parameter allows for
            ignoring zero paddings.

      Returns:
        'nmsed_boxes': A [max_detections, 4] float32 tensor containing the
          non-max suppressed boxes.
        'nmsed_scores': A [max_detections] float32 tensor containing the scores
          for the boxes.
        'nmsed_classes': A [max_detections] float32 tensor containing the class
          for boxes.
        'num_detections': A [batch_size] int32 tensor indicating the number of
          valid detections per batch item. Only the top num_detections[i]
          entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The
          rest of the entries are zero paddings.
      """
      per_image_boxes = args[0]
      per_image_scores = args[1]
      per_image_clip_window = args[2]
      per_image_num_valid_boxes = args[-1]
      per_image_boxes = tf.reshape(
          tf.slice(per_image_boxes, 3 * [0],
                   tf.stack([per_image_num_valid_boxes, -1, -1])), [-1, q, 4])
      per_image_scores = tf.reshape(
          tf.slice(per_image_scores, [0, 0],
                   tf.stack([per_image_num_valid_boxes, -1])),
          [-1, num_classes])
      nmsed_boxlist, num_valid_nms_boxes = multiclass_non_max_suppression(
          per_image_boxes,
          per_image_scores,
          score_thresh,
          iou_thresh,
          max_size_per_class,
          max_total_size,
          clip_window=per_image_clip_window)

      if not use_static_shapes:
        nmsed_boxlist = box_list_ops.pad_or_clip_box_list(
            nmsed_boxlist, max_total_size)
      num_detections = num_valid_nms_boxes
      nmsed_boxes = nmsed_boxlist.get()
      nmsed_scores = nmsed_boxlist.get_field("scores")
      nmsed_classes = nmsed_boxlist.get_field("classes")
      return ([nmsed_boxes, nmsed_scores, nmsed_classes, num_detections])

    batch_outputs = shape_utils.static_or_dynamic_map_fn(
        _single_image_nms_fn,
        elems=([boxes, scores, clip_window, num_valid_boxes]),
        dtype=(3 * [tf.float32] + [tf.int32]),
        parallel_iterations=parallel_iterations)

    batch_nmsed_boxes = batch_outputs[0]
    batch_nmsed_scores = batch_outputs[1]
    batch_nmsed_classes = batch_outputs[2]
    batch_num_detections = batch_outputs[-1]

    return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes, batch_num_detections)
