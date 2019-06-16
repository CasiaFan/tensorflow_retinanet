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
"""RetinaNet feature extractors based on Resnet v1.

See https://arxiv.org/abs/1708.02002 for details.
"""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.retinanet import retinanet_fpn

RESNET_ARCH_BLOCK = {"resnet50": [3, 4, 6, 3],
                     "resnet101": [3, 4, 23, 3]}

class RetinaNetFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD FPN feature extractor based on Resnet v1 architecture."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               backbone,
               fpn_scope_name,
               min_level=3,
               max_level=7,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """RetinaNet feature extractor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      fpn_scope_name: scope name under which to construct the feature pyramid
        network.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.

    Raises:
      ValueError: On supplying invalid arguments for unused arguments.
    """
    super(RetinaNetFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        conv_hyperparams_fn=conv_hyperparams_fn,
        pad_to_multiple=pad_to_multiple,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    if self._use_explicit_padding is True:
      raise ValueError('Explicit padding is not a valid option.')
    self._backbone = backbone
    self._fpn_scope_name = fpn_scope_name
    self._min_level = min_level
    self._max_level = max_level
    self._additional_layer_depth = additional_layer_depth

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      return resized_inputs - [[channel_means]]
    else:
      return resized_inputs

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)
    with tf.variable_scope(
        self._fpn_scope_name, reuse=self._reuse_weights) as scope:
        if self._backbone in list(RESNET_ARCH_BLOCK.keys()):
          block_layers = RESNET_ARCH_BLOCK[self._backbone]
        else:
          raise ValueError("Unknown backbone found! Only resnet50 or resnet101 is allowed!")
        image_features = retinanet_fpn(inputs=preprocessed_inputs, 
                                       block_layers=block_layers,
                                       depth=self._additional_layer_depth,
                                       is_training=self._is_training)
        return [image_features[x] for x in range(self._min_level, sefl._max_level+1)]

class RetinaNet50FeatureExtractor(RetinaNetFeatureExtractor):
  """Resnet 50 RetinaNet feature extractor."""
  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               backbone='resnet50',
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(RetinaNet50FeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        conv_hyperparams_fn,
        pad_to_multiple,
        'resnet50',
        'retinanet50',
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

class RetinaNet101FeatureExtractor(RetinaNetFeatureExtractor):
  """Resnet 101 RetinaNet feature extractor."""
  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               backbone='resnet101',
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(RetinaNet101FeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        conv_hyperparams_fn,
        pad_to_multiple,
        'resnet101',
        'retinanet101',
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)