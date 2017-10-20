from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from fpn import RetinaNet_FPN101
import tensorflow as tf

slim = tf.contrib.slim

class RetinaNet():
    """ RetinaNet defined in Focal loss paper
     See: https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self):
        self._retina_fpn = RetinaNet_FPN101

    def __call__(self, inputs, num_classes, num_anchors=9, scope=None, reuse=None):
        """
        Args:
            num_classes: # of classification classes
            num_anchors: # of anchors in each feature map
        """
        self._num_classes = num_classes
        self._num_anchors = num_anchors
        self._scope = scope
        self._reuse = reuse
        return self._forward(inputs)

    def _add_fcn_head(self, inputs, output_planes, head_offset):
        """
        inputs: a [batch, height, width, channels] float tensor
        output_planes: # of outputs dim
        layer_offset: idx of feature maps
        """
        with tf.variable_scope(self._scope, "Retina_FCN_Head_"+str(head_offset), [inputs], reuse=self._reuse):
            net = slim.repeat(inputs, 4, slim.conv2d, 256, kernel_size=[3, 3], activation_fn=tf.nn.relu)
            net = slim.conv2d(net, output_planes, kernel_size=[3, 3], activation_fn=None)
        return net

    def _forward(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_maps = self._retina_fpn(inputs)
        loc_predictions = []
        class_predictions = []
        for idx, feature_map in enumerate(feature_maps):
            loc_prediction = self._add_fcn_head(feature_map,
                                                self._num_anchors * 4,
                                                "Box")
            class_prediction = self._add_fcn_head(feature_map,
                                                  self._num_anchors*self._num_classes,
                                                  "Class")
            loc_prediction = tf.reshape(loc_prediction, [batch_size, -1, 4])
            class_prediction = tf.reshape(class_prediction, [batch_size, -1, self._num_classes])
            loc_predictions.append(loc_prediction)
            class_predictions.append(class_prediction)
        return tf.concat(loc_predictions, axis=1), tf.concat(class_predictions, axis=1)


def test():
    net = RetinaNet()
    inputs = tf.Variable(tf.random_normal([2, 224, 224, 3], dtype=tf.float32), name="inputs")
    loc_predictions, class_predictions = net(inputs, 20, 9)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run([loc_predictions, class_predictions]))
        print(sess.run(tf.shape(loc_predictions)))
        print(sess.run(tf.shape(class_predictions)))
    sess.close()

if __name__ == "__main__":
    test()