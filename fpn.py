"""Contains definition of FPN architecture.

As described by Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He et.al
    Feature Pyramid Networks for Object Detection. arXiv: 1612.03144

FPN: input shape [batch, 224, 224, 3]
    with slim.arg_scope(fpn.fpn_arg_scope(is_training)):
        net, endpoints = fpn.fpn101(inputs,
                                    blocks=[2, 4, 23, 3],
                                    is_training=False)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import resnet_utils as ru
slim = tf.contrib.slim

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               rate=1, scope=None):
    """Bottleneck residual unit variant with BN before convolutions
    When putting together 2 consecutive ResNet blocks that use this unit,
    one should use stride =2 in the last unit of first block

    NOTE: This scripts refer to keras resnet50
    Args:
        inputs: A tensor of size [batchsize, height, width, channels] (after BN)
        depth: The depth of the ResNet unit output
        depth_bottleneck: The depth of bottleneck layers
        stride: the ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input
        scope: Optional variable_scope

    Returns:
        The ResNet unit output
    """
    with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # shortcut
        if depth == depth_in:
            # identity block with no conv layer at shortcut
            shortcut = ru.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=1, scope='shortcut')
        # layer1
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1', normalizer_fn=None,
                               activation_fn=None)
        # layer 2
        residual = ru.conv2d_same(residual, depth_bottleneck, 3, stride=stride,
                                    rate=rate, scope='conv2')
        # layer 3
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, scope='conv3',
                               normalizer_fn=None, activation_fn=None)

        output = shortcut + residual
        return output


def resnet_v2_block(scope, base_depth, num_planes, stride):
    """
    Args:
        scope: The scope of the block
        base_depth: The depth of bottleneck layer for each unit
        num_planes: the number of planes in the block
        stride: The stride of the block, implemented as a stride in the last unit
          All other stride is 1

    Returns:
        A resnet_v2 bottleneck block object
    """
    return ru.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1}] * (num_planes - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


def stack_resnet_v2_units(inputs, block):
    """ Stack ResNet planes

    This function creates scopes for the ResNet in the form of
    'block_name/plane_1, block_name/plane_2', etc.

    Most ResNets consists of 4 ResNet blocks and subsample the activations by
    a factor of 2 when transitioning between consecutive ResNet blocks.

    Args:
        inputs: a Tensor of size [batch, height, width, channels]
        block: A list of ResNet block object describing the units in the block

    Returns:
        output: A tensor with stride equal to the specified stride in the block
            and same batch_size shape
    """
    with tf.variable_scope(block.scope, 'block', [inputs]):
        net = inputs
        for i, unit in enumerate(block.args):
            with tf.variable_scope('unit_%d' %(i+1), values=[inputs]):
                net = block.unit_fn(net, **unit)
    return net


def FPN(inputs,
        num_planes,
        num_channels=256,
        is_training=True,
        reuse=None,
        scope=None):
    """ Generator for FPN models.

    At bottom up stage, FPN use ResNet as backbone and feature activation outputs by each stages last residual block.
    By default, 4 blocks are used in ResNest: {C2, C3, C4, C5} with {4, 8, 16, 32} strides with
    respect to input image.

    At top down stage, with a coarser-resolution feature map, up-sample it by factor 2 then mergeed
    with corresponding bottom up layer (which undergoes a 1x1 conv to reduce dimension)
    by element-wise addition, called {P2, P3, P4, P5}.
    Attach a 1x1 conv layer on C5 to produce coarsest resolution map, then finally append 3x3 conv
    on each merged map to reduce alias effect of up-sampling. Because all levels of pyramid
    use shared classifier, feature dimension (output channel) is fixed to d=256.

    NOTE: P6 is simply a stride 2 sub-sampling of P5, for covering a coarser anchor scale of 512^2
    Args:
        inputs: A tensor of size [batchsize, height, width, channels]
        num_planes: A list of of length equal to the number of ResNet blocks. Each
            element is the number of planes in each ResNet block.
        num_channels: The number of output feature channels
        is_training: whether is training or not
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given
        scope: variable scope

    Returns:
        5 feature map tensors: {P2, P3, P4, P5, P6}
    """
    with tf.variable_scope(scope, 'FPN', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, bottleneck]):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                c1 = ru.conv2d_same(inputs, 64, 7, stride=2, scope='conv1')
                bn1 = slim.batch_norm(c1, scope='norm1', activation_fn=tf.nn.relu)
                mp1 = slim.max_pool2d(bn1, [3, 3], stride=2, scope='pool1', padding='SAME')
                # Bottom up
                # block 1, down-sampling is done in conv3_1, conv4_1, conv5_1
                block1 = resnet_v2_block('block1', base_depth=64, num_planes=num_planes[0], stride=1)
                c2 = stack_resnet_v2_units(mp1, block1)
                # block 2
                block2 = resnet_v2_block('block2', base_depth=128, num_planes=num_planes[1], stride=2)
                c3 = stack_resnet_v2_units(c2, block2)
                # block 3
                block3 = resnet_v2_block('block3', base_depth=256, num_planes=num_planes[2], stride=2)
                c4 = stack_resnet_v2_units(c3, block3)
                # block 4
                block4 = resnet_v2_block('block4', base_depth=512, num_planes=num_planes[3], stride=2)
                c5 = stack_resnet_v2_units(c4, block4)

                # lateral layer
                l2 = slim.conv2d(c2, num_channels, [1, 1], stride=1, scope='lat2')
                l3 = slim.conv2d(c3, num_channels, [1, 1], stride=1, scope='lat3')
                l4 = slim.conv2d(c4, num_channels, [1, 1], stride=1, scope='lat4')
                p5 = slim.conv2d(c5, num_channels, [1, 1], stride=1, scope='lat5')

                # Top down
                t4 = slim.conv2d_transpose(c5, num_channels, [4, 4], stride=[2, 2])
                p4 = ru.conv2d_same(t4+l4, num_channels, 3, stride=1)
                t3 = slim.conv2d_transpose(p4, num_channels, [4, 4], stride=[2, 2])
                p3 = ru.conv2d_same(t3+l3, num_channels, 3, stride=1)
                t2 = slim.conv2d_transpose(p3, num_channels, [4, 4], stride=[2, 2])
                p2 = ru.conv2d_same(t2+l2, num_channels, 3, stride=1)
                return p2, p3, p4, p5
FPN.default_image_size = 600  # shorter side in COCO image set

def FPN50(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 4, 6, 3]
    return FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
FPN50.default_image_size = FPN.default_image_size


def FPN101(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 4, 23, 3]
    return FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
FPN101.default_image_size = FPN.default_image_size

def FPN152(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 8, 36, 3]
    return FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
FPN152.default_image_size = FPN.default_image_size

def FPN200(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 24, 36, 3]
    return FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
FPN200.default_image_size = FPN.default_image_size


def RetinaNet_FPN(inputs,
        num_planes,
        num_channels=256,
        is_training=True,
        reuse=None,
        scope=None):
    """ Generator for RetinaNet FPN models. A small modification of initial FPN model for returning layers
        {P3, P4, P5, P6, P7}. See paper Focal Loss for Dense Object Detection. arxiv: 1708.02002

        P2 is discarded and P6 is obtained via 3x3 stride-2 conv on c5; P7 is computed by applying ReLU followed by
        3x3 stride-2 conv on P6. P7 is to improve large object detection
    Args:
        same as FPN module

    Returns:
        5 feature map tensors: {P3, P4, P5, P6, P7}
    """
    with tf.variable_scope(scope, 'Retina_FPN', [inputs], reuse=reuse) as sc:
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, bottleneck]):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    c1 = ru.conv2d_same(inputs, 64, 7, stride=2, scope='conv1')
                    bn1 = slim.batch_norm(c1, scope='norm1', activation_fn=tf.nn.relu)
                    mp1 = slim.max_pool2d(bn1, [3, 3], stride=2, scope='pool1', padding='SAME')
                    # Bottom up
                    # block 1, down-sampling is done in conv3_1, conv4_1, conv5_1
                    block1 = resnet_v2_block('block1', base_depth=64, num_planes=num_planes[0], stride=1)
                    c2 = stack_resnet_v2_units(mp1, block1)
                    # block 2
                    block2 = resnet_v2_block('block2', base_depth=128, num_planes=num_planes[1], stride=2)
                    c3 = stack_resnet_v2_units(c2, block2)
                    # block 3
                    block3 = resnet_v2_block('block3', base_depth=256, num_planes=num_planes[2], stride=2)
                    c4 = stack_resnet_v2_units(c3, block3)
                    # block 4
                    block4 = resnet_v2_block('block4', base_depth=512, num_planes=num_planes[3], stride=2)
                    c5 = stack_resnet_v2_units(c4, block4)
                    # P6
                    p6 = ru.conv2d_same(c5, num_channels, 3, stride=2, scope='conv6')
                    # P7
                    p7 = ru.conv2d_same(tf.nn.relu(p6), num_channels, 3, stride=2, scope='conv7')

                    # lateral layer
                    l3 = slim.conv2d(c3, num_channels, [1, 1], stride=1, scope='lat3')
                    l4 = slim.conv2d(c4, num_channels, [1, 1], stride=1, scope='lat4')
                    p5 = slim.conv2d(c5, num_channels, [1, 1], stride=1, scope='conv5')
                    # Top down
                    t4 = slim.conv2d_transpose(p5, num_channels, [4, 4], stride=[2, 2])
                    p4 = ru.conv2d_same(t4+l4, num_channels, 3, stride=1)
                    t3 = slim.conv2d_transpose(p4, num_channels, [4, 4], stride=[2, 2])
                    p3 = ru.conv2d_same(t3+l3, num_channels, 3, stride=1)
    return p3, p4, p5, p6, p7
FPN.default_image_size = 512  # shorter side in COCO image set


def RetinaNet_FPN50(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 4, 6, 3]
    return RetinaNet_FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
RetinaNet_FPN50.default_image_size = FPN.default_image_size


def RetinaNet_FPN101(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 4, 23, 3]
    return RetinaNet_FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
RetinaNet_FPN101.default_image_size = FPN.default_image_size

def RetinaNet_FPN152(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 8, 36, 3]
    return RetinaNet_FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
RetinaNet_FPN152.default_image_size = FPN.default_image_size

def RetinaNet_FPN200(inputs,
          is_training=True,
          reuse=None,
          scope=None):
    num_planes = [3, 24, 36, 3]
    return RetinaNet_FPN(inputs,
               num_planes=num_planes,
               is_training=is_training,
               reuse=reuse,
               scope=scope)
RetinaNet_FPN200.default_image_size = FPN.default_image_size


def test():
    inputs = tf.Variable(tf.random_normal([2, 224, 224, 3]), name='inputs')
    fms = FPN101(inputs)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for fm in fms:
            print(fm)
    sess.close()

if __name__ == "__main__":
    test()