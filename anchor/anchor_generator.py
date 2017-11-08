from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from anchor import box_list_ops
from anchor import box_list

import tensorflow as tf

class MultipleGridAnchorGenerator():
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               box_specs_list,
               base_anchor_sizes,
               clip_window=None):
    """Constructs a MultipleGridAnchorGenerator.

    To construct anchors, at multiple grid resolutions, one must provide a
    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid
    size, a corresponding list of (scale, aspect ratio) box specifications.

    For example:
    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid
                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid

    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while scale and aspect ratios are fixed at construction
    time.

    Args:
      box_specs_list: list of list of (scale, aspect ratio) pairs with the
        outside list having the same number of entries as feature_map_shape_list
        (which is passed in at generation time).
      base_anchor_sizes: list of base anchor size in each layer
      clip_window: a tensor of shape [4] specifying a window to which all
        anchors should be clipped. If clip_window is None, then no clipping
        is performed.

    Raises:
      ValueError: if box_specs_list is not a list of list of pairs
      ValueError: if clip_window is not either None or a tensor of shape [4]
    """
    if isinstance(box_specs_list, list) and all(
        [isinstance(list_item, list) for list_item in box_specs_list]):
      self._box_specs = box_specs_list
    else:
      raise ValueError('box_specs_list is expected to be a '
                       'list of lists of pairs')
    if isinstance(base_anchor_sizes, list):
        self._base_anchor_sizes = base_anchor_sizes
    else:
        raise ValueError('base_anchor_list is expected to be a list of float')
    if clip_window is not None and clip_window.get_shape().as_list() != [4]:
      raise ValueError('clip_window must either be None or a shape [4] tensor')
    self._clip_window = clip_window
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      if not all([isinstance(entry, tuple) and len(entry) == 2
                  for entry in box_spec]):
        raise ValueError('box_specs_list is expected to be a '
                         'list of lists of pairs')
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

  def name_scope(self):
    return 'MultipleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return [len(box_specs) for box_specs in self._box_specs]

  def generate(self,
                input_size,
                feature_map_shape_list,
                anchor_strides=None,
                anchor_offsets=None):
    """Generates a collection of bounding boxes to be used as anchors.

    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids. In our box_specs_list example
    (see the constructor docstring), we would place two boxes over each grid
    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
    output anchors follows the order of how the grid sizes and box_specs are
    specified (with box_spec index varying the fastest, followed by width
    index, then height index, then grid index).

    Args:
      input_size: input image size list with (width, height)
      feature_map_shape_list: list of pairs of conv net layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      anchor_strides: list of pairs of strides (in y and x directions
        respectively). For example, setting
        anchor_strides=[(.25, .25), (.5, .5)] means that we want the anchors
        corresponding to the first layer to be strided by .25 and those in the
        second layer to be strided by .5 in both y and x directions. By
        default, if anchor_strides=None, then they are set to be the reciprocal
        of the corresponding grid sizes. The pairs can also be specified as
        dynamic tf.int or tf.float numbers, e.g. for variable shape input
        images.
      anchor_offsets: list of pairs of offsets (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offsets=[(.125, .125), (.25, .25)]) means that we want the
        (0, 0)-th anchor of the first layer to lie at (.125, .125) in image
        space and likewise that we want the (0, 0)-th anchor of the second
        layer to lie at (.25, .25) in image space. By default, if
        anchor_offsets=None, then they are set to be half of the corresponding
        anchor stride. The pairs can also be specified as dynamic tf.int or
        tf.float numbers, e.g. for variable shape input images.

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == len(self._box_specs)):
      raise ValueError('feature_map_shape_list must be a list with the same '
                       'length as self._box_specs')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    im_height, im_width = input_size[0], input_size[1]
    if not anchor_strides:
      anchor_strides = [(tf.to_float(im_height) / tf.to_float(pair[0]),
                         tf.to_float(im_width) / tf.to_float(pair[1]))
                        for pair in feature_map_shape_list]
    if not anchor_offsets:
      anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                        for stride in anchor_strides]
    for arg, arg_name in zip([anchor_strides, anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if not all([isinstance(list_item, tuple) and len(list_item) == 2
                  for list_item in arg]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

    anchor_grid_list = []
    for grid_size, scales, aspect_ratios, stride, offset, base_anchor_size in zip(
        feature_map_shape_list, self._scales, self._aspect_ratios,
        anchor_strides, anchor_offsets, self._base_anchor_sizes):
      anchor_grid_list.append(
          tile_anchors(
              grid_height=grid_size[0],
              grid_width=grid_size[1],
              scales=scales,
              aspect_ratios=aspect_ratios,
              base_anchor_size=base_anchor_size,
              anchor_stride=stride,
              anchor_offset=offset))
    concatenated_anchors = box_list_ops.concatenate(anchor_grid_list)
    num_anchors = concatenated_anchors.num_boxes_static()
    if num_anchors is None:
      num_anchors = concatenated_anchors.num_boxes()
    if self._clip_window is not None:
      clip_window = tf.multiply(
          tf.to_float([im_height, im_width, im_height, im_width]),
          self._clip_window)
      concatenated_anchors = box_list_ops.clip_to_window(
          concatenated_anchors, clip_window, filter_nonoverlapping=False)
      # TODO: make reshape an option for the clip_to_window op
      concatenated_anchors.set(
          tf.reshape(concatenated_anchors.get(), [num_anchors, 4]))

    stddevs_tensor = 0.01 * tf.ones(
        [num_anchors, 4], dtype=tf.float32, name='stddevs')
    concatenated_anchors.add_field('stddev', stddevs_tensor)
    return concatenated_anchors


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size in this layer as [height, width]
        (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size
  widths = scales * ratio_sqrts * base_anchor_size
  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
  widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  # convert [ycenter, xcenter, height, width] to [ymin, xmin, ymax, xmax]
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  return box_list.BoxList(bbox_corners)


def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


def create_retinanet_anchors(
                       num_layers=5,
                       scales=(1.0, pow(2, 1./3), pow(2, 2./3)),
                       aspect_ratios=(0.5, 1.0, 2.0),
                       base_anchor_sizes=(32.0, 64.0, 128.0, 256.0, 512.0)):
    """Create a set of anchors walking along a grid in a collection of feature maps in RetinaNet.

    This op creates a set of anchor boxes by placing a basis collection of
    boxes with user-specified scales and aspect ratios centered at evenly
    distributed points along a grid. The basis  Each box is multiplied by
    base_anchor_size before placing it over its respective center.

    Args:
        num_layers: The number of grid layers to create anchors
        scales: A list of scales
        aspect_ratios: A list of aspect ratios
        base_anchor_sizes: List of base anchor sizes in each layer
    Returns:
        A MultipleGridAnchorGenerator
    """
    base_anchor_sizes = list(base_anchor_sizes)
    box_spec_list = []
    for idx in range(num_layers):
        layer_spec_list = []
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                layer_spec_list.append((scale, aspect_ratio))
        box_spec_list.append(layer_spec_list)
    return MultipleGridAnchorGenerator(box_spec_list, base_anchor_sizes)


def anchor_assign(anchors, gt_boxes, gt_labels, is_training=True):
    """
    Assign generated anchors to boxes and labels
    Args:
        anchors: BoxList holding a collection of N anchors
        gt_boxes: BoxList holding a collection of groundtruth 2D box coordinates tensor/list [#object, 4]
            ([ymin, xmin, xmax, ymax], float type) of objects in given input image.
        gt_labels: Groundtruth 1D tensor/list [#object] (scalar int) of objects in given image.
        is_training: is training or not

    returns:
        BoxList with anchor location and class fields
    """
    pos_iou_thred = 0.5
    neg_iou_thred = 0.5
    if is_training:
        neg_iou_thred = 0.4

    box_iou = box_list_ops.iou(anchors, gt_boxes)
    # get max iou ground truth box of each anchor
    anchor_max_iou = tf.reduce_max(box_iou, axis=1)
    anchor_max_iou_indices = tf.argmax(box_iou, axis=1)
    anchor_gt_box = tf.gather(gt_boxes.get(), anchor_max_iou_indices)
    # add background category
    anchor_gt_cls = tf.gather(gt_labels, anchor_max_iou_indices)
    # get remaining index with iou between 0.4 to 0.5
    # modify ignored labels to -1 and background labels to 0
    anchor_gt_cls = tf.where(tf.greater(anchor_max_iou, pos_iou_thred), anchor_gt_cls, 0-tf.ones_like(anchor_gt_cls))
    anchor_gt_cls = tf.where(tf.less(anchor_max_iou, neg_iou_thred), tf.zeros_like(anchor_gt_cls), anchor_gt_cls)
    anchors.add_field('gt_boxes', anchor_gt_box)
    anchors.add_field('gt_labels', anchor_gt_cls)
    return anchors


def test():
    input_size = [224, 224]
    feature_maps = [(tf.ceil(input_size[0]/pow(2., i+3)), tf.ceil(input_size[1]/pow(2., i+3))) for i in range(5)]
    anchor_generator = create_retinanet_anchors()
    anchors = anchor_generator.generate(input_size, feature_maps)
    gt_boxes = box_list.BoxList(tf.convert_to_tensor([[10, 10, 150, 150], [30, 40, 60, 70]], dtype=tf.float32))
    gt_labels = tf.convert_to_tensor([1, 2])
    anchors = anchor_assign(anchors, gt_boxes, gt_labels)
    x = tf.convert_to_tensor([[[1,2,3],[3,4,5],[5,6,7]],[[1,2,3],[3,4,5],[5,6,7]]])
    with tf.Session() as sess:
        result = anchors.get_field("gt_boxes")
        # result = anchors.get()
        print(sess.run(result))
        print(sess.run(tf.shape(result)))
        print(sess.run(tf.squeeze(tf.where(tf.greater(gt_labels, 1)))))
        print(sess.run(tf.gather(x, tf.convert_to_tensor([0,1]), axis=1)))
    sess.close()

if __name__ == "__main__":
    test()