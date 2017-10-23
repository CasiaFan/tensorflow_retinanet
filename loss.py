import tensorflow as tf

slim = tf.contrib.slim

def focal_loss_with_logits(onehot_labels, logits,
                            alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute softmax focal loss between logits and onehot labels

    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)

    Args:
      labels: Each row labels[i] must be a valid probability distribution
      logits: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)

    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [logits, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(logits)
        onehot_labels = tf.convert_to_tensor(onehot_labels)

        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        predictions = tf.nn.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        epsilon = 1e-14
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        losses = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * onehot_labels * tf.log(predictions_pt+epsilon),
                                     name=name)
        return losses


def regression_loss(pred_boxes, gt_boxes):
    """
    Regression loss (Smooth L1 loss: also known as huber loss)
    """
    loss = tf.losses.huber_loss(predictions=pred_boxes, labels=gt_boxes,
                                weights=1.0, scope='box_loss')
    return loss


def test():
    logits = tf.convert_to_tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]])
    labels = slim.one_hot_encoding([1, 2], 4)
    bbox = tf.ones_like(logits)
    with tf.Session() as sess:
        print sess.run(logits)
        print sess.run(focal_loss_with_logits(onehot_labels=labels, logits=logits))
        print sess.run(regression_loss(logits, bbox))
    sess.close()

# test()