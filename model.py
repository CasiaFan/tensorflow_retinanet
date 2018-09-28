import tensorflow as tf
from object_detection import learning_schedules


DEFAULT_PARAMS = tf.contrib.training.HParams(
                  image_size=640,
                  num_classes=90,
                  num_scales=3,
                  aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                  anchor_scale=4.0,
                  box_loss_weight=50.0)


def learning_rate_schedule(total_steps):
    """
    Use cosine decay with warmup learning rate schedule. See parameters from tensorflow object detection api.
    """
    learning_rate = learning_schedules.cosine_decay_with_warmup(global_step=tf.train.get_or_create_global_step(),
                                                                learning_rate_base=0.04,
                                                                total_steps=total_steps,
                                                                warmup_learning_rate=0.13333,
                                                                warmup_steps=2000)
    return learning_rate


def model_fn(features, labels, mode, params=DEFAULT_PARAMS):
    """
    Model definition for estimator framework
    """

