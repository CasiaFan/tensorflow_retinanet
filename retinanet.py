"""Contains definition of RetinaNet architecture.

As described by Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He et.al
    Feature Pyramid Networks for Object Detection. arXiv: 1612.03144

FPN: input shape [batch, 224, 224, 3]
    with slim.arg_scope(fpn.fpn_arg_scope(is_training)):
        net, endpoints = fpn.fpn101(inputs,
                                    blocks=[2, 4, 23, 3],
                                    is_training=False)
"""
import tensorflow as tf
from object_detection.shape_utils import combined_static_and_dynamic_shape

BN_PARAMS = {"bn_decay": 0.997,
             "bn_epsilon": 0.001}


# define number of layers of each block for different architecture
RESNET_ARCH_BLOCK = {"resnet50": [3, 4, 6, 3],
                     "resnet101": [3, 4, 23, 3]}


def nearest_neighbor_upsampling(input_tensor, scale):
    """Nearest neighbor upsampling implementation.
    NOTE: See TensorFlow Object Detection API uitls.ops
    Args:
        input_tensor: A float32 tensor of size [batch, height_in, width_in, channels].
        scale: An integer multiple to scale resolution of input data.
    Returns:
        upsample_input: A float32 tensor of size [batch, height_in*scale, width_in*scale, channels].
    """
    with tf.name_scope('nearest_neighbor_upsampling'):
        (batch_size, h, w, c) = combined_static_and_dynamic_shape(input_tensor)
        output_tensor = tf.reshape(input_tensor, [batch_size, h, 1, w, 1, c]) * tf.ones(
                [1, 1, scale, 1, scale, 1], dtype=input_tensor.dtype)
        return tf.reshape(output_tensor, [batch_size, h*scale, w*scale, c])


def conv2d_same(inputs, depth, kernel_size, strides, scope=None):
    with tf.name_scope(scope, None):
        if strides == 1:
            return tf.layers.conv2d(inputs, depth, kernel_size, padding='SAME')
        else:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return tf.layers.conv2d(inputs,
                                    depth,
                                    kernel_size,
                                    strides=strides,
                                    padding='VALID',
                                    use_bias=False,
                                    kernel_initializer=tf.variance_scaling_initializer())


def _bn(inputs, is_training, name=None):
    return tf.layers.batch_normalization(inputs,
                                         training=is_training,
                                         momentum=BN_PARAMS["bn_decay"],
                                         epsilon=BN_PARAMS["bn_epsilon"],
                                         scale=True,
                                         name=name)


def bottleneck(inputs, depth, strides, is_training, projection=False, scope=None):
    """Bottleneck residual unit variant with BN after convolutions
       When putting together 2 consecutive ResNet blocks that use this unit,
       one should use stride =2 in the last unit of first block

    Args:
        inputs: A tensor of size [batchsize, height, width, channels] (after BN)
        depth: The depth of the block unit output
        strides: the ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input
        is_training: indicate training state for BN layer
        projection: if this block will use a projection. True for first block in block groups
        scope: Optional variable scope

    Returns:
        The ResNet unit output
    """
    with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
        # shortcut connection
        shortcut = inputs
        depth_out = depth * 4
        if projection:
            shortcut = conv2d_same(shortcut, depth_out, kernel_size=1, strides=strides, scope='shortcut')
            shortcut = _bn(shortcut, is_training)

        # layer1
        residual = conv2d_same(inputs, depth, kernel_size=1, strides=1, scope='conv1')
        residual = _bn(residual, is_training)
        residual = tf.nn.relu6(residual)
        # layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=strides, scope='conv2')
        residual = _bn(residual, is_training)
        residual = tf.nn.relu6(residual)
        # layer 3
        residual = conv2d_same(residual, depth_out, kernel_size=1, strides=1, scope='conv3')
        residual = _bn(residual, is_training)
        output = shortcut + residual
        return tf.nn.relu6(output)


def stack_bottleneck(inputs, layers, depth, strides, is_training, scope=None):
    """ Stack bottleneck planes

    This function creates scopes for the ResNet in the form of 'block_name/plane_1, block_name/plane_2', etc.
    Args:
        layers: number of layers in this block
    """
    with tf.variable_scope(scope, 'block', [inputs]) as sc:
        inputs = bottleneck(inputs, depth, strides=strides, is_training=is_training, projection=True)
        for i in range(1, layers):
            layer_scope = "unit_{}".format(i)
            inputs = bottleneck(inputs, depth, strides=1, is_training=is_training, scope=layer_scope)
    return inputs


def retinanet_fpn(inputs,
                  block_layers,
                  depth=256,
                  is_training=True,
                  scope=None):
    """
    Generator for RetinaNet FPN models. A small modification of initial FPN model for returning layers
        {P3, P4, P5, P6, P7}. See paper Focal Loss for Dense Object Detection. arxiv: 1708.02002

        P2 is discarded and P6 is obtained via 3x3 stride-2 conv on c5; P7 is computed by applying ReLU followed by
        3x3 stride-2 conv on P6. P7 is to improve large object detection

    Returns:
        5 feature map tensors: {P3, P4, P5, P6, P7}
    """
    with tf.variable_scope(scope, 'retinanet_fpn', [inputs]) as sc:
        net = conv2d_same(inputs, 64, kernel_size=7, strides=2, scope='conv1')
        net = _bn(net, is_training)
        net = tf.nn.relu6(net)
        net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='SAME', name='pool1')

        # Bottom up
        # block 1, down-sampling is done in conv3_1, conv4_1, conv5_1
        p2 = stack_bottleneck(net, layers=block_layers[0], depth=64, strides=1, is_training=is_training)
        # block 2
        p3 = stack_bottleneck(p2, layers=block_layers[1], depth=128, strides=2, is_training=is_training)
        # block 3
        p4 = stack_bottleneck(p3, layers=block_layers[2], depth=256, strides=2, is_training=is_training)
        # block 4
        p5 = stack_bottleneck(p4, layers=block_layers[3], depth=512, strides=2, is_training=is_training)
        p5 = tf.identity(p5, name="p5")
        # coarser FPN feature
        # p6
        p6 = tf.layers.conv2d(p5, filters=depth, kernel_size=3, strides=2, name='conv6', padding='SAME')
        p6 = _bn(p6, is_training)
        p6 = tf.nn.relu6(p6)
        p6 = tf.identity(p6, name="p6")
        # P7
        p7 = tf.layers.conv2d(p6, filters=depth, kernel_size=3, strides=2, name='conv7', padding='SAME')
        p7 = _bn(p7, is_training)
        p7 = tf.identity(p7, name="p7")

        # lateral layer
        l3 = tf.layers.conv2d(p3, filters=depth, kernel_size=1, strides=1, name='l3', padding='SAME')
        l4 = tf.layers.conv2d(p4, filters=depth, kernel_size=1, strides=1, name='l4', padding='SAME')
        l5 = tf.layers.conv2d(p5, filters=depth, kernel_size=1, strides=1, name='l5', padding='SAME')
        # Top down
        t4 = nearest_neighbor_upsampling(l5, 2) + l4
        p4 = tf.layers.conv2d(t4, filters=depth, kernel_size=3, strides=1, name='t4', padding='SAME')
        p4 = _bn(p4, is_training)
        p4 = tf.identity(p4, name="p4")
        t3 = nearest_neighbor_upsampling(t4, 2) + l3
        p3 = tf.layers.conv2d(t3, filters=depth, kernel_size=3, strides=1, name='t3', padding='SAME')
        p3 = _bn(p3, is_training)
        p3 = tf.identity(p3, name="p3")
        features = {3: p3,
                    4: p4,
                    5: l5,
                    6: p6,
                    7: p7}
        return features


def share_weight_class_net(inputs, level, num_classes, num_anchors_per_loc, num_layers_before_predictor=4, is_training=True):
    """
    net for predicting class labels
    NOTE: Share same weights when called more then once on different feature maps
    Args:
        inputs: feature map with shape (batch_size, h, w, channel)
        level: which feature map
        num_classes: number of predicted classes
        num_anchors_per_loc: number of anchors at each spatial location in feature map
        num_layers_before_predictor: number of the additional conv layers before the predictor.
        is_training: is in training or not
    returns:
        feature with shape (batch_size, h, w, num_classes*num_anchors)
    """
    for i in range(num_layers_before_predictor):
        inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, strides=1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  padding="SAME",
                                  name='class_{}'.format(i))
        inputs = _bn(inputs, is_training, name="class_{}_bn_level_{}".format(i, level))
        inputs = tf.nn.relu(inputs)
    outputs = tf.layers.conv2d(inputs,
                               filters=num_classes*num_anchors_per_loc,
                               kernel_size=3,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               padding="SAME",
                               name="class_pred")
    return outputs


def share_weight_box_net(inputs, level, num_anchors_per_loc, num_layers_before_predictor=4, is_training=True):
    """
    Similar to class_net with output feature shape (batch_size, h, w, num_anchors*4)
    """
    for i in range(num_layers_before_predictor):
        inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, strides=1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  padding="SAME",
                                  name='box_{}'.format(i))
        inputs = _bn(inputs, is_training, name="box_{}_bn_level_{}".format(i, level))
        inputs = tf.nn.relu6(inputs)
    outputs = tf.layers.conv2d(inputs,
                               filters=4*num_anchors_per_loc,
                               kernel_size=3,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               padding="SAME",
                               name="box_pred")
    return outputs


def retinanet(images, num_classes, num_anchors_per_loc, resnet_arch='resnet50', is_training=True):
    """
    Get box prediction features and class prediction features from given images
    Args:
        images: input batch of images with shape (batch_size, h, w, 3)
        num_classes: number of classes for prediction
        num_anchors_per_loc: number of anchors at each feature map spatial location
        resnet_arch: name of which resnet architecture used
        is_training: indicate training or not
    return:
        prediciton dict: holding following items:
            box_predictions tensor from each feature map with shape (batch_size, num_anchors, 4)
            class_predictions_with_bg tensor from each feature map with shape (batch_size, num_anchors, num_class+1)
            feature_maps: list of tensor of feature map
    """
    assert resnet_arch in list(RESNET_ARCH_BLOCK.keys()), "resnet architecture not defined"
    with tf.variable_scope('retinanet'):
        batch_size = combined_static_and_dynamic_shape(images)[0]
        features = retinanet_fpn(images, block_layers=RESNET_ARCH_BLOCK[resnet_arch], is_training=is_training)
        class_pred = []
        box_pred = []
        feature_map_list = []
        num_slots = num_classes + 1
        with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
            for level in features.keys():
                class_outputs = share_weight_class_net(features[level], level,
                                                       num_slots,
                                                       num_anchors_per_loc,
                                                       is_training=is_training)
                class_outputs = tf.reshape(class_outputs, shape=[batch_size, -1, num_slots])
                class_pred.append(class_outputs)
                feature_map_list.append(features[level])
        with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
            for level in features.keys():
                box_outputs = share_weight_box_net(features[level], level, num_anchors_per_loc, is_training=is_training)
                box_outputs = tf.reshape(box_outputs, shape=[batch_size, -1, 4])
                box_pred.append(box_outputs)
        return dict(box_pred=tf.concat(box_pred, axis=1),
                    cls_pred=tf.concat(class_pred, axis=1),
                    feature_map_list=feature_map_list)
