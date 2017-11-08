from retinanet import RetinaNet
from loss import focal_loss, regression_loss
from utils.preprocess import preprocess
from utils import model_deploy
from model_config import config
from anchor.anchor_generator import create_retinanet_anchors, anchor_assign
from anchor.box_coder import FasterRCNNBoxCoder
from data_generator.input_generator import input_queue_generator
from data_generator.dataset_util import get_label_map_dict

import tensorflow as tf
from functools import partial

slim = tf.contrib.slim

def _get_inputs(input_queue, num_classes, is_training=True):
  """Dequeue batch and construct inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.

  Returns:
    images: a list of 3-D float tensor of images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    anchor_list: a list of anchors containing anchors
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 0  # class index starts from 1, 0 means background
  # anchor encoder
  anchor_encoder = FasterRCNNBoxCoder()
  def extract_images_and_targets(read_data):
    image = read_data['image']
    location_gt = read_data['groundtruth_boxes']
    classes_gt = tf.cast(read_data['groundtruth_classes'],
                         tf.int32)
    classes_gt -= label_id_offset
    # add background padding
    classes_onehot = tf.one_hot(classes_gt, depth=num_classes)
    bg_pad = tf.cast(tf.where(tf.greater_equal(classes_gt, 0), tf.zeros_like(classes_gt)), tf.ones_like(classes_gt))
    classes_onehot = tf.concat([bg_pad, classes_onehot], axis=1)
    # get anchors
    anchors_list = []
    for i in range(tf.shape(classes_gt)[0]):
        # get anchors
        input_size = tf.shape(image[i])
        feature_map_list = [(tf.ceil(input_size[0]/pow(2., i+3)), tf.ceil(input_size[1]/pow(2., i+3))) for i in range(5)]
        anchor_generator = create_retinanet_anchors()
        anchor = anchor_generator.geznerate(input_size, feature_map_list)
        anchor = anchor_assign(anchor, gt_boxes=location_gt[i], gt_labels=classes_gt[i], is_training=is_training)
        # encode anchor boxes
        gt_boxes = anchor.get_field('gt_boxes')
        encoded_gt_boxes = anchor_encoder.encode(gt_boxes, anchor.get())
        anchor.set_field('encoded_gt_boxes', encoded_gt_boxes)
        anchors_list.append(anchor)
    return image, location_gt, classes_onehot, anchors_list
  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, num_classes):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: num of classes, integer
  Returns:
    Average sum of loss of given input batch samples with shape
  """
  (images, groundtruth_boxes_list, groundtruth_classes_list, anchors_list) = _get_inputs(input_queue, num_classes)
  images = [preprocess(image, im_width=config.MODEL.IM_WIDTH,
                       im_height=config.MODEL.IM_HEIGHT,
                       preprocess_options=config.TRAIN.PREPROCESS) for image in images]
  images = tf.concat(images, 0)
  net = RetinaNet()
  loc_preds, cls_preds = net(images, num_classes+1, anchors=9)
  # get num of anchor overlapped with ground truth box
  cls_gt = [anchor.get_field("gt_labels") for anchor in anchors_list]
  loc_gt = [anchor.get_field("gt_encoded_boxes") for anchor in anchors_list]
  # pos anchor count for each image
  gt_anchor_nums = tf.map_fn(lambda x: tf.reduce_sum(tf.cast(tf.greater(x, 0), tf.int32)), cls_gt)
  # get valid anchor indices
  valid_anchor_indices = tf.squeeze(tf.where(tf.greater_equal(cls_gt, 0)))
  # skip ignored anchors (iou belong to 0.4 to 0.5)
  [valid_cls_preds, valid_cls_gt] = map(lambda x: tf.gather(x, valid_anchor_indices, axis=1),
                                           [cls_preds, cls_gt])
  # classification loss: convert to onehot code
  cls_loss = tf.multiply(focal_loss(valid_cls_gt, valid_cls_preds), 1./tf.to_float(gt_anchor_nums))
  # location regression loss
  valid_cls_indices = tf.squeeze(tf.where(tf.greater(cls_gt, 0)))
  # skip negative and ignored anchors
  [valid_loc_preds, valid_loc_gt] = map(lambda x: tf.gather(x, valid_cls_indices, axis=1),
                                           [loc_preds, loc_gt])
  loc_loss = regression_loss(valid_loc_preds, valid_loc_gt, weights=tf.expand_dims(1./tf.to_float(gt_anchor_nums), 1))
  loss = (tf.reduce_sum(loc_loss) + tf.reduce_sum(cls_loss)) / tf.size(gt_anchor_nums, out_type=tf.float32)
  return loss


def train(train_config, train_dir, master, task=0,
          num_clones=1, worker_replicas=1, clone_on_cpu=False, ps_tasks=0, worker_job_name='lonely_worker',
          is_chief=True):
  """Training function for detection models.

  Args:
    train_config: configuration of parameters for model training.
    train_dir: Directory to write checkpoints and training summaries to.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
  """

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    with tf.device(deploy_config.inputs_device()):
      train_config.batch_size = train_config.batch_size // num_clones
      input_queue = input_queue_generator(train_config)
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    # get num of classes
    num_classes = get_label_map_dict(train_config.TRAIN.label_map_file)
    model_fn = partial(_create_losses, num_classes)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                   global_summaries)

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint)
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer, regularization_losses=None)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    gpu_memory_fraction = 0.8
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        sync_optimizer=sync_optimizer,
        saver=saver)