import tensorflow as tf

slim = tf.contrib.slim

def focal_loss(onehot_labels, cls_preds,
                            alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute sigmoid focal loss between logits and onehot labels

    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)

    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)

    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(precise_logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        epsilon = 1e-8
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt+epsilon),
                                     name=name, axis=1)
        return losses


def regression_loss(pred_boxes, gt_boxes, weights):
    """
    Regression loss (Smooth L1 loss: also known as huber loss)

    Args:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    """
    loss = tf.losses.huber_loss(predictions=pred_boxes, labels=gt_boxes,
                                weights=weights, scope='box_loss')
    return loss


def test():
    logits = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]])
    labels = slim.one_hot_encoding([1, 2], 4)
    bbox = tf.ones_like(logits)
    with tf.Session() as sess:
        print sess.run(logits)
        print sess.run(focal_loss(onehot_labels=labels, cls_preds=logits))
        print sess.run(regression_loss(logits, bbox, tf.expand_dims(1./tf.convert_to_tensor([2, 3], dtype=tf.float32), 1)))
    sess.close()

# test()