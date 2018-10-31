import tensorflow as tf


def focal_loss(logits, onehot_labels, weights, alpha=0.25, gamma=2.0):
    """
    Compute sigmoid focal loss between logits and onehot labels: focal loss = -(1-pt)^gamma*log(pt)

    Args:
        onehot_labels: onehot labels with shape (batch_size, num_anchors, num_classes)
        logits: last layer feature output with shape (batch_size, num_anchors, num_classes)
        weights: weight tensor returned from target assigner with shape [batch_size, num_anchors]
        alpha: The hyperparameter for adjusting biased samples, default is 0.25
        gamma: The hyperparameter for penalizing the easy labeled samples, default is 2.0
    Returns:
        a scalar of focal loss of total classification
    """
    with tf.name_scope("focal_loss"):
        logits = tf.cast(logits, tf.float32)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
        predictions = tf.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        weighted_loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t * tf.expand_dims(weights, axis=2)
        return tf.reduce_sum(weighted_loss)


def regression_loss(pred_boxes, gt_boxes, weights, delta=1.0):
    """
    Regression loss (Smooth L1 loss: also known as huber loss)

    Args:
        pred_boxes: [batch_size, num_anchors, 4]
        gt_boxes: [batch_size, num_anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [batch_size, num_anchors]
        delta: delta for smooth L1 loss
    Returns:
        a box regression loss scalar
    """
    loss = tf.reduce_sum(tf.losses.huber_loss(predictions=pred_boxes,
                                              labels=gt_boxes,
                                              delta=delta,
                                              weights=tf.expand_dims(weights, axis=2),
                                              scope='box_loss',
                                              reduction=tf.losses.Reduction.NONE))
    return loss


onehot_labels = tf.constant([[1 ,0], [1, 0]], tf.float32)
logits = tf.constant([[290, -500], [10, -2]], tf.float32)
ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
with tf.Session() as sess:
    print(sess.run(ce))