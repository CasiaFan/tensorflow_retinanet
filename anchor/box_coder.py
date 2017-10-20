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

"""box coder following Faster RCNN procedure.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow as tf

from anchor import box_list

EPSILON = 1e-8

def batch_decode(encoded_boxes, box_coder, anchors):
  """Decode a batch of encoded boxes.

  This op takes a batch of encoded bounding boxes and transforms
  them to a batch of bounding boxes specified by their corners in
  the order of [y_min, x_min, y_max, x_max].

  Args:
    encoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      code_size] representing the location of the objects.
    box_coder: a BoxCoder object.
    anchors: a BoxList of anchors used to encode `encoded_boxes`.

  Returns:
    decoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      coder_size] representing the corners of the objects in the order
      of [y_min, x_min, y_max, x_max].

  Raises:
    ValueError: if batch sizes of the inputs are inconsistent, or if
    the number of anchors inferred from encoded_boxes and anchors are
    inconsistent.
  """
  encoded_boxes.get_shape().assert_has_rank(3)
  if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
    raise ValueError('The number of anchors inferred from encoded_boxes'
                     ' and anchors are inconsistent: shape[1] of encoded_boxes'
                     ' %s should be equal to the number of anchors: %s.' %
                     (encoded_boxes.get_shape()[1].value,
                      anchors.num_boxes_static()))

  decoded_boxes = tf.stack([
      box_coder.decode(boxes, anchors).get()
      for boxes in tf.unstack(encoded_boxes)
  ])
  return decoded_boxes


class FasterRCNNBoxCoder():
  """Faster RCNN box coder."""

  def __init__(self, scale_factors=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 4
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  def encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))

  def decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))

