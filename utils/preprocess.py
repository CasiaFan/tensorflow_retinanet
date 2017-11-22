
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

"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputDataFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""


import tensorflow as tf


def _get_default_func_arg_map():
  """Returns the default mapping from a preprocessor function to its args.

  Args:
    include_instance_masks: If True, preprocessing functions will modify the
      instance masks, too.
    include_keypoints: If True, preprocessing functions will modify the
      keypoints, too.

  Returns:
    A map from preprocessing functions to the arguments they receive.
  """
  prep_func_arg_map = {
      normalize_image: ('image',),
      random_horizontal_flip: ('image',
                               'groundtruth_boxes'),
      random_vertical_flip: ('image',
                             'groundtruth_boxes'),
      random_pixel_value_scale: ('image',),
      random_image_scale: ('image'),
      random_rgb_to_gray: ('image',),
      random_adjust_brightness: ('image',),
      random_adjust_contrast: ('image',),
      random_adjust_hue: ('image',),
      random_adjust_saturation: ('image',),
      random_distort_color: ('image',),
      random_jitter_boxes: ('groundtruth_boxes',),
      flip_boxes_horizontally: ('groundtruth_boxes',),
      flip_boxes_vertically: ('groundtruth_boxes',),
  }
  return prep_func_arg_map


imagenet_mean_value = [123.68, 116.779, 103.939]  # RGB
def _preprocess_pixel_value(image):
    image = tf.to_float(image)
    image = subtract_channel_mean(image, imagenet_mean_value)
    image = tf.multiply(image, 2./255.)
    return image


def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.

  Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.to_float(image)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) /
                        (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image


def subtract_channel_mean(image, means=None):
  """Normalizes an image by subtracting a mean from each channel.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    means: float list containing a mean for each channel
  Returns:
    normalized_images: a tensor of shape [height, width, channels]
  Raises:
    ValueError: if images is not a 4D tensor or if the number of means is not
      equal to the number of channels.
  """
  with tf.name_scope('SubtractChannelMean', values=[image, means]):
    if len(image.get_shape()) != 3:
      raise ValueError('Input must be of size [height, width, channels]')
    if len(means) != image.get_shape()[-1]:
      raise ValueError('len(means) must match the number of channels')
    return image - [[means]]


def resize_image(image,
                 new_height=600,
                 new_width=1024,
                 method=tf.image.ResizeMethod.BILINEAR):
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method]):
    result = tf.image.resize_images(image, [new_height, new_width], method=method)
    return result


def flip_boxes_horizontally(boxes):
  """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Horizontally flipped boxes.
  """
  # Flip boxes horizontally.
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def flip_boxes_vertically(boxes):
    """Up-down flip the boxes

    Args:
      boxes: rank 2 float32 tensor containing bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1]
             Each row is in the form of [ymin, xmin, ymax, xmax]
    Returns:
      Vertically flipped boxes
    """
    # Flip boxes vertically
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_ymin = tf.subtract(1.0, ymax)
    flipped_ymax = tf.subtract(1.0, ymin)
    flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], axis=1)
    return flipped_boxes


def random_horizontal_flip(
    image,
    boxes=None,
    seed=None):
  """Randomly decides whether to horizontally mirror the image and detections or not.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """
  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.random_uniform([], seed=seed)
    # flip only if there are bounding boxes in image!
    do_a_flip_random = tf.logical_and(
        tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    result.append(image)

    # flip boxes
    if boxes is not None:
      boxes = tf.cond(
          do_a_flip_random, lambda: flip_boxes_horizontally(boxes), lambda: boxes)
      result.append(boxes)

    return tuple(result)


def random_vertical_flip(
    image,
    boxes=None,
    seed=None):
  """Randomly decides whether to vertically mirror the image and detections or not.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """
  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_up_down(image)
    return image_flipped

  with tf.name_scope('RandomVerticalFlip', values=[image, boxes]):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.random_uniform([], seed=seed)
    # flip only if there are bounding boxes in image!
    do_a_flip_random = tf.logical_and(
        tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    result.append(image)

    # flip boxes
    if boxes is not None:
      boxes = tf.cond(
          do_a_flip_random, lambda: flip_boxes_vertically(boxes), lambda: boxes)
      result.append(boxes)

    return tuple(result)

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
  """Scales each value in the pixels of the image.

     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomPixelValueScale', values=[image]):
    color_coef = tf.random_uniform(
        tf.shape(image),
        minval=minval,
        maxval=maxval,
        dtype=tf.float32,
        seed=seed)
    image = tf.multiply(image, color_coef)
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

def random_image_scale(image,
                       masks=None,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
  """Scales the image size.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
    masks: (optional) rank 3 float32 tensor containing masks with
      size [height, width, num_masks]. The value is set to None if there are no
      masks.
    min_scale_ratio: minimum scaling ratio.
    max_scale_ratio: maximum scaling ratio.
    seed: random seed.

  Returns:
    image: image which is the same rank as input image.
    masks: If masks is not none, resized masks which are the same rank as input
      masks will be returned.
  """
  with tf.name_scope('RandomImageScale', values=[image]):
    result = []
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    size_coef = tf.random_uniform([],
                                  minval=min_scale_ratio,
                                  maxval=max_scale_ratio,
                                  dtype=tf.float32, seed=seed)
    image_newysize = tf.to_int32(
        tf.multiply(tf.to_float(image_height), size_coef))
    image_newxsize = tf.to_int32(
        tf.multiply(tf.to_float(image_width), size_coef))
    image = tf.image.resize_images(
        image, [image_newysize, image_newxsize], align_corners=True)
    result.append(image)
    if masks:
      masks = tf.image.resize_nearest_neighbor(
          masks, [image_newysize, image_newxsize], align_corners=True)
      result.append(masks)
    return tuple(result)


def random_rgb_to_gray(image, probability=0.1, seed=None):
  """Changes the image from RGB to Grayscale with the given probability.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """
  def _image_to_gray(image):
    image_gray1 = tf.image.rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    # random variable defining whether to do flip or not
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image,
        lambda: _image_to_gray(image))

  return image

def random_adjust_brightness(image, max_delta=0.2):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).

  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    image = tf.image.random_brightness(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    image = tf.image.random_contrast(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_hue(image, max_delta=0.02):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    image = tf.image.random_hue(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    image = tf.image.random_saturation(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.

  Randomly distorts color using a combination of brightness, hue, contrast
  and saturation changes. Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).

  Returns:
    image: image which is the same shape as input image.

  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_jitter_boxes(boxes, ratio=0.05, seed=None):
  """Randomly jitter boxes in image.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    ratio: The ratio of the box width and height that the corners can jitter.
           For example if the width is 100 pixels and ratio is 0.05,
           the corners can jitter up to 5 pixels in the x direction.
    seed: random seed.

  Returns:
    boxes: boxes which is the same shape as input boxes.
  """
  def random_jitter_box(box, ratio, seed):
    """Randomly jitter box.

    Args:
      box: bounding box [1, 1, 4].
      ratio: max ratio between jittered box and original box,
      a number between [0, 0.5].
      seed: random seed.

    Returns:
      jittered_box: jittered box.
    """
    rand_numbers = tf.random_uniform(
        [1, 1, 4], minval=-ratio, maxval=ratio, dtype=tf.float32, seed=seed)
    box_width = tf.subtract(box[0, 0, 3], box[0, 0, 1])
    box_height = tf.subtract(box[0, 0, 2], box[0, 0, 0])
    hw_coefs = tf.stack([box_height, box_width, box_height, box_width])
    hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
    jittered_box = tf.add(box, hw_rand_coefs)
    jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
    return jittered_box

  with tf.name_scope('RandomJitterBoxes', values=[boxes]):
    # boxes are [N, 4]. Lets first make them [N, 1, 1, 4]
    boxes_shape = tf.shape(boxes)
    boxes = tf.expand_dims(boxes, 1)
    boxes = tf.expand_dims(boxes, 2)

    distorted_boxes = tf.map_fn(
        lambda x: random_jitter_box(x, ratio, seed), boxes, dtype=tf.float32)

    distorted_boxes = tf.reshape(distorted_boxes, boxes_shape)

    return distorted_boxes


preprocess_options_map = {
    'normalize_image': (normalize_image, {'original_minval': 0,
                        'original_maxval': 255,
                        'target_minval': 0,
                        'target_maxval': 1}),
    'random_horizontal_flip': (random_horizontal_flip, {}),
    'random_vertical_flip': (random_vertical_flip, {}),
    'subtract_channel_mean': (subtract_channel_mean, {}),
    'random_pixel_value_scale': (random_pixel_value_scale, {}),
    'random_image_scale': (random_image_scale, {}),
    'random_rgb_to_gray': (random_rgb_to_gray, {}),
    'random_adjust_brightness': (random_adjust_brightness, {}),
    'random_adjust_contrast': (random_adjust_contrast, {}),
    'random_adjust_hue': (random_adjust_hue, {}),
    'random_adjust_saturation': (random_adjust_saturation, {}),
    'random_distort_color': (random_distort_color, {}),
    'random_jitter_boxes': (random_jitter_boxes, {})
}

def preprocess(tensor_dict, preprocess_options, im_height=224, im_width=224):
  """Preprocess images and bounding boxes.

  Various types of preprocessing (to be implemented) based on the
  preprocess_options dictionary e.g. "crop image" (affects image and possibly
  boxes), "white balance image" (affects only image), etc. If self._options
  is None, no preprocessing is done.

  Args:
    tensor_dict: dictionary that contains images, boxes, and can contain other
                 things as well.
                 images-> rank 4 float32 tensor contains
                          1 image -> [1, height, width, 3].
                          with pixel values varying between [0, 1]
                 boxes-> rank 2 float32 tensor containing
                         the bounding boxes -> [N, 4].
                         Boxes are in normalized form meaning
                         their coordinates vary between [0, 1].
                         Each row is in the form
                         of [ymin, xmin, ymax, xmax].
    preprocess_options: It is a list each contains a function name.
  Returns:
    tensor_dict: which contains the preprocessed images, bounding boxes, etc.

  Raises:
    ValueError: (a) If the functions passed to Preprocess
                    are not in func_arg_map.
                (b) If the arguments that a function needs
                    do not exist in tensor_dict.
                (c) If image in tensor_dict is not rank 4
  """
  # mapping from preprocessing functions to arguments
  func_arg_map = _get_default_func_arg_map()
  # changes the images to image (rank 4 to rank 3) since the functions
  # receive rank 3 tensor for image
  if 'image' in tensor_dict:
    images = tensor_dict['image']
    if len(images.get_shape()) != 4:
      raise ValueError('images in tensor_dict should be rank 4')
    image = tf.squeeze(images, squeeze_dims=[0])
    # resize image
    image = resize_image(image, new_height=im_height, new_width=im_width)
    # preprocess pixels
    image = _preprocess_pixel_value(image)
    tensor_dict['image'] = image

  # Preprocess inputs based on preprocess_options
  for option in preprocess_options:
    func, params = preprocess_options_map[option]
    if func not in func_arg_map:
      raise ValueError('The function %s does not exist in func_arg_map' %
                       (func.__name__))
    arg_names = func_arg_map[func]
    for a in arg_names:
      if a is not None and a not in tensor_dict:
        raise ValueError('The function %s requires argument %s' %
                         (func.__name__, a))

    def get_arg(key):
      return tensor_dict[key] if key is not None else None
    args = [get_arg(a) for a in arg_names]
    results = func(*args, **params)
    if not isinstance(results, (list, tuple)):
      results = (results,)
    # Removes None args since the return values will not contain those.
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]
    for res, arg_name in zip(results, arg_names):
      tensor_dict[arg_name] = res

  # changes the image to images (rank 3 to rank 4) to be compatible to what
  # we received in the first place
  if 'image' in tensor_dict:
    image = tensor_dict['image']
    images = tf.expand_dims(image, 0)
    tensor_dict['image'] = images

  return tensor_dict