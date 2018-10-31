import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from object_detection import learning_schedules
from object_detection import post_processing
from object_detection import shape_utils
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection.label_map_util import create_categories_from_labelmap
from object_detection.coco_evaluation import CocoDetectionEvaluator
from anchors import create_target_assigner, batch_assign_targets, Anchor
from retinanet import retinanet
from loss import focal_loss, regression_loss


DEFAULT_PARAMS = tf.contrib.training.HParams(
                  num_scales=2,
                  aspect_ratios=(1.0, 2.0, 0.5),
                  anchor_scale=4.0,
                  box_loss_weight=50.0,
                  total_train_steps=200000,
                  iou_thres=0.6,
                  score_thres=0.1,
                  max_detections_per_class=100,
                  max_detections_total=100,
                  learning_rate=0.08,
                  momentum=0.9)


def learning_rate_schedule(total_steps):
    """
    Use cosine decay with warmup learning rate schedule. See parameters from tensorflow object detection api.
    """
    learning_rate = learning_schedules.cosine_decay_with_warmup(global_step=tf.train.get_or_create_global_step(),
                                                                learning_rate_base=0.04,
                                                                total_steps=total_steps,
                                                                warmup_learning_rate=0.0133,
                                                                warmup_steps=100)
    return learning_rate


"""Refer to ssd_meta_arch in tensorflow object detection api"""
class RetinaNetModel():
    """RetinaNet mode constructor"""
    def __init__(self, is_training, num_classes, params=DEFAULT_PARAMS):
        """
        Args:
            is_training: indicate training or not
            num_classes: number of classes for prediction
            params: parameters for model definition
                    resnet_arch: name of which resnet architecture used
        """
        self._is_training = is_training
        self._num_classes = num_classes
        self._nms_fn = post_processing.batch_multiclass_non_max_suppression
        self._score_convert_fn = tf.sigmoid
        self._params = params
        # self._unmatched_class_label = tf.constant([1] + (self._num_classes) * [0], tf.float32)
        self._unmatched_class_label = tf.constant((self._num_classes + 1) * [0], tf.float32)
        self._target_assigner = create_target_assigner(unmatched_cls_target=self._unmatched_class_label)
        self._anchors = None
        self._anchor_generator = None
        self._box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    @property
    def anchors(self):
        if not self._anchors:
            raise RuntimeError("anchors have not been constructed yet")
        return self._anchors

    def _get_feature_map_shape(self, features):
        """Return list of spatial dimensions for each feature map"""
        feature_map_shapes = [shape_utils.combined_static_and_dynamic_shape(feature) for feature in features]
        return [(shape[1], shape[2]) for shape in feature_map_shapes]

    def predict(self, inputs):
        """
        Perform predict from batched input tensor.
        During this time, anchors must be constructed before post-process or loss function called
        Args:
            inputs: a [batch_size, height, width, channels] image tensor
        Returns:
            prediction_dict: dict with items:
                inputs: [batch_size, height, width, channels] image tensor
                box_pred: [batch_size, num_anchors, 4] tensor containing predicted boxes
                cls_pred: [batch_size, num_anchors, num_classes+1] tensor containing class predictions
                feature_maps: a list of feature map tensor
                anchors: [num_anchors, 4] tensor containing anchors in normalized coordinates
        """
        num_anchors_per_loc = self._params.get("num_scales") * len(self._params.get("aspect_ratios"))
        prediction_dict = retinanet(inputs, self._num_classes, num_anchors_per_loc, is_training=self._is_training)
        # generate anchors
        feature_map_shape_list = self._get_feature_map_shape(prediction_dict["feature_map_list"])
        image_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
        # initialize anchor generator
        if self._anchor_generator is None:
            self._anchor_generator = Anchor(feature_map_shape_list=feature_map_shape_list,
                                            img_size=(image_shape[1], image_shape[2]),
                                            anchor_scale=self._params.get("anchor_scale"),
                                            aspect_ratios=self._params.get("aspect_ratios"),
                                            scales_per_octave=self._params.get("num_scales"))
        self._anchors = self._anchor_generator.boxes
        prediction_dict["inputs"] = inputs
        prediction_dict["anchors"] = self._anchors
        return prqediction_dict

    def _batch_decode(self, box_encodings):
        """
        Decode batch of box encodings with respect to anchors
        Args:
            box_encodings: box prediction tensor with shape [batch_size, num_anchors, 4]
        Returns:
            decoded_boxes: decoded box tensor with same shape as input tensor
        """
        input_shape = shape_utils.combined_static_and_dynamic_shape(box_encodings)
        batch_size = input_shape[0]
        tiled_anchor_boxes = tf.tile(tf.expand_dims(self._anchors, 0), [batch_size, 1, 1])
        tiled_anchor_boxlist = box_list.BoxList(tf.reshape(tiled_anchor_boxes, [-1, 4]))
        decoded_boxes = self._box_coder.decode(tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
                                               tiled_anchor_boxlist)
        return tf.reshape(decoded_boxes.get(), [batch_size, -1, 4])

    def preprocess(self, inputs, img_size):
        image = tf.image.resize_images(inputs, size=(img_size, img_size))
        image = (image - 128.0) / 255.0 * 2
        return image

    def postprocess(self, prediction_dict, score_thres=1e-8):
        """
        Convert prediction tensors to final detection by slicing the bg class, decoding box predictions,
                applying nms and clipping to image window
        Args:
            prediction_dict: dict returned by self.predict function
            score_thres: threshold for score to remove low confident boxes
        Returns:
            detections: a dict with these items:
                detection_boxes: [batch_size, max_detection, 4]
                detection_scores: [batch_size, max_detections]
                detection_classes: [batch_size, max_detections]
        """
        with tf.name_scope('Postprocessor'):
            box_pred = prediction_dict["box_pred"]
            cls_pred = prediction_dict["cls_pred"]
            # decode box
            detection_boxes = self._batch_decode(box_pred)
            detection_boxes = tf.expand_dims(detection_boxes, axis=2)
            # sigmoid function to calculate score from feature
            detection_scores_with_bg = tf.sigmoid(cls_pred, name="converted_scores")
            # slice detection scores without score
            detection_scores = tf.slice(detection_scores_with_bg, [0, 0, 1], [-1, -1, -1])
            clip_window = tf.constant([0, 0, 1, 1], dtype=tf.float32)
            (nms_boxes, nms_scores, nms_classes,
             num_detections) = post_processing.batch_multiclass_non_max_suppression(detection_boxes,
                                                                                    detection_scores,
                                                                                    score_thresh=score_thres,
                                                                                    iou_thresh=self._params.get("iou_thres"),
                                                                                    max_size_per_class=self._params.get("max_detections_per_class"),
                                                                                    max_total_size=self._params.get("max_detections_total"),
                                                                                    clip_window=clip_window)
            return dict(detection_boxes=nms_boxes,
                        detection_scores=nms_scores,
                        detection_classes=nms_classes,
                        num_detections=num_detections)

    def _assign_targets(self, gt_boxes_list, gt_labels_list):
        """
        Assign gt targets
        Args:
             gt_boxes_list: a list of 2-D tensor of shape [num_boxes, 4] containing coordinates of gt boxes
             gt_labels_list: a list of 2-D one-hot tensors of shape [num_boxes, num_classes] containing gt classes
        Returns:
            batch_cls_targets: class tensor with shape [batch_size, num_anchors, num_classes]
            batch_reg_target: box tensor with shape [batch_size, num_anchors, 4]
            match_list: a list of matcher.Match object encoding the match between anchors and gt boxes for each image
                        of the batch, with rows corresponding to gt-box and columns corresponding to anchors
        """
        gt_boxlist_list = [box_list.BoxList(boxes) for boxes in gt_boxes_list]
        gt_labels_with_bg = [tf.pad(gt_class, [[0, 0], [1, 0]], mode='CONSTANT')
                              for gt_class in gt_labels_list]
        anchors = box_list.BoxList(self._anchors)
        return batch_assign_targets(self._target_assigner,
                                    anchors,
                                    gt_boxlist_list,
                                    gt_labels_with_bg)

    def _summarize_target_assignment(self, gt_boxes_list, match_list):
        """
        Create tf summaries for input box and anchors: average num of 1) gt-boxes; 2) anchors marked as pos;
                3) anchors marked as neg; 4) anchors marked as ignored
        Args:
            gt_boxes_list: a list of 2-D box tensors of shape [num_anchros, 4]
            match_list: a list of matcher.Match object returned by target_assigner function
        """
        num_boxes_per_image = tf.stack([tf.shape(x)[0] for x in gt_boxes_list])
        pos_anchors_per_image = tf.stack([match.num_matched_columns() for match in match_list])
        neg_anchors_per_image = tf.stack([match.num_unmatched_columns() for match in match_list])
        ignored_anchors_per_image = tf.stack([match.num_ignored_columns() for match in match_list])
        tf.summary.scalar('AvgNumGroundtruthBoxesPerImage', tf.reduce_mean(tf.to_float(num_boxes_per_image)),
                          family='TargetAssignment')
        tf.summary.scalar('AvgNumPositiveAnchorsPerImage', tf.reduce_mean(tf.to_float(pos_anchors_per_image)),
                          family='TargetAssignment')
        tf.summary.scalar('AvgNumNegativeAnchorsPerImage', tf.reduce_mean(tf.to_float(neg_anchors_per_image)),
                          family='TargetAssignment')
        tf.summary.scalar('AvgNumIgnoredAnchorsPerImage', tf.reduce_mean(tf.to_float(ignored_anchors_per_image)),
                          family='TargetAssignment')

    def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
        def _add_cdf_image_summary(values, name):
            def cdf_plot(values):
                # numpy function to plot CDF
                normalized_values = values / np.sum(values)
                sorted_values = np.sort(normalized_values)
                cum_values = np.cumsum(sorted_values)
                fraction_of_examples = (np.arange(cum_values.size, dtype=np.float32) / cum_values.size)
                fig = plt.figure(frameon=False)
                ax = fig.add_subplot('111')
                ax.plot(fraction_of_examples, cum_values)
                ax.set_ylabel('cumulative normalized values')
                ax.set_xlabel('fraction of examples')
                fig.canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
                return image
            cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
            tf.summary.image(name,cdf_plot)
        pos_indices = tf.where(tf.greater(class_ids, 0))
        pos_anchor_cls_loss = tf.squeeze(tf.gather(cls_losses, pos_indices), axis=1)
        _add_cdf_image_summary(pos_anchor_cls_loss, 'PositiveAnchorLossCDF')
        neg_indices = tf.where(tf.equal(class_ids, 0))
        neg_anchor_cls_loss = tf.squeeze(tf.gather(cls_losses, neg_indices), axis=1)
        _add_cdf_image_summary(neg_anchor_cls_loss, 'NegativeAnchorLossCDF')

    def loss(self, prediction_dict, gt_boxes_list, gt_labels_list):
        """
        Compute loss between prediction tensor and gt
        Args:
            prediction_dict: dict of following items
                box_encodings: a [batch_size, num_anchors, 4] containing predicted boxes
                cls_pred_with_bg: a [batch_size, num_anchors, num_classes+1] containing predicted classes
            gt_boxes_list: a list of 2D gt box tensor with shape [num_boxes, 4]
            gt_labels_list: a list of 2-D gt one-hot class tensor with shape [num_boxes, num_classes]
        Returns:
            a dictionary with localization_loss and classification_loss
        """
        with tf.name_scope(None, 'Loss', prediction_dict.values()):
            (batch_cls_targets, batch_cls_weights, batch_reg_targets, batch_reg_weights,
                match_list) = self._assign_targets(gt_boxes_list, gt_labels_list)
            # num_positives = [tf.reduce_sum(tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
            #                      for matches in match_list]
            self._summarize_target_assignment(gt_boxes_list, match_list)
            reg_loss = regression_loss(prediction_dict["box_pred"], batch_reg_targets, batch_reg_weights)
            cls_loss = focal_loss(prediction_dict["cls_pred"], batch_cls_targets, batch_cls_weights)
            # normalize loss by num of matches
            # num_pos_anchors = [tf.reduce_sum(tf.cast(tf.not_equal(match.match_results, -1), tf.float32))
            #                    for match in match_list]
            normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)), 1.0)
            # normalize reg loss by box codesize (here is 4)
            reg_normalizer = normalizer * 4
            normalized_reg_loss = tf.multiply(reg_loss, 1.0/reg_normalizer, name="regression_loss")
            normalized_cls_loss = tf.multiply(cls_loss, 1.0/normalizer, name="classification_loss")
            return normalized_reg_loss, normalized_cls_loss, batch_reg_weights, batch_cls_weights


    def restore_map(self):
        variables_to_restore = {}
        for variable in tf.global_variables():
            var_name = variable.op.name
            if var_name.startswith("retinanet"):
                variables_to_restore[var_name] = variable
        return variables_to_restore


def unstack_batch(tensor_dict):
    """
    Unstack input tensor along 0th dimension
    Args:
        tensor_dict: dict of tensor with shape (batch_size, num_boxes, d1, .., dn), including:
            gt_labels, gt_boxes, num_gt_boxes
    """
    # # extract tensor from tuple. TODO: figure out where box tuple comes from?
    for key in tensor_dict.keys():
        if key == "gt_boxes":
            tensor_dict["gt_boxes"] = tensor_dict["gt_boxes"][0]
    unbatched_tensor_dict = {key: tf.unstack(tensor) for key, tensor in tensor_dict.items()}
    # remove padding along 'num_boxes' dimension of the gt tensors
    num_gt_list = unbatched_tensor_dict["num_gt_boxes"]
    unbatched_unpadded_tensor_dict = {}
    for key in unbatched_tensor_dict:
        if key == "num_gt_boxes":
            continue
        unpadded_tensor_list = []
        for num_gt, padded_tensor in zip(num_gt_list, unbatched_tensor_dict[key]):
            tensor_shape = shape_utils.combined_static_and_dynamic_shape(padded_tensor)
            slice_begin = tf.zeros(len(tensor_shape), dtype=tf.int32)
            slice_size = tf.stack([num_gt] + [-1 if dim is None else dim for dim in tensor_shape[1:]])
            unpadded_tensor = tf.slice(padded_tensor, slice_begin, slice_size)
            unpadded_tensor_list.append(unpadded_tensor)
        unbatched_unpadded_tensor_dict[key] = unpadded_tensor_list
    return unbatched_unpadded_tensor_dict


def _get_variables_available_in_ckpt(variables, ckpt_path):
    """
    Return the subset of available in the ckpt
    """
    ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_dtype_map()
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for var_name, variable in sorted(variables.items()):
        if var_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[var_name] == variable.shape.as_list():
                vars_in_ckpt[var_name] = variable
    return vars_in_ckpt


def _scale_to_abs_coord(boxes, y_scale, x_scale):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(boxes, num_or_size_splits=4, axis=1)
    y_min *= y_scale
    x_min *= x_scale
    y_max *= y_scale
    x_max *= x_scale
    return tf.concat([y_min, x_min, y_max, x_max], 1)


def _result_dict_for_single_example(image, key, detections, gt_boxes, gt_labels):
    """
    Merge all detection and gt information for a single example
    """
    image_shape = tf.shape(image)
    detection_boxes = detections["detection_boxes"][0]
    detection_scores = detections["detection_scores"][0]
    detection_classes = tf.to_int64(detections["detection_classes"][0])
    num_detections = tf.to_int32(detections["num_detections"][0])
    detection_boxes = tf.slice(detection_boxes, begin=[0, 0], size=[num_detections, -1])
    detection_classes = tf.slice(detection_classes, begin=[0], size=[num_detections])
    detection_scores = tf.slice(detection_scores, begin=[0], size=[num_detections])
    # scale to abs coordinates
    y_scale, x_scale = image_shape[1], image_shape[2]
    detection_boxes = _scale_to_abs_coord(detection_boxes, y_scale, x_scale)
    gt_boxes = _scale_to_abs_coord(gt_boxes, y_scale, x_scale)
    return dict(image=image,
                key=key,
                detection_boxes=detection_boxes,
                detection_scores=detection_scores,
                detection_classes=detection_classes,
                num_detections=num_detections,
                groundtruth_boxes=gt_boxes,
                groundtruth_classes=gt_labels)


def create_model_fn(run_config, default_params=DEFAULT_PARAMS):
    def model_fn(features, labels, mode, params):
        """
        Model definition for estimator framework
        Args:
            features: dict of input image tensor with shape (batch_size, height, width, 3)
            labels: dict of input labels
        Return:
            EstimatorSpec to run training, evaluation, or prediction
        """
        label_offset = 1
        total_loss, train_op, detections, export_outputs = None, None, None, None
        is_training = mode==tf.estimator.ModeKeys.TRAIN
        num_classes = run_config["num_classes"]
        model = RetinaNetModel(is_training=is_training, num_classes=num_classes)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # load pretrained model for checkpoint
            ckpt_file = run_config.get("finetune_ckpt")
            if ckpt_file:
                asg_map = model.restore_map()
                available_var_map = (_get_variables_available_in_ckpt(asg_map, ckpt_file))
                tf.train.init_from_checkpoint(ckpt_file, available_var_map)
        # predict
        images = features["image"]
        keys = features["key"]
        predictions_dict = model.predict(images)
        # postprocess
        if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            detections = model.postprocess(predictions_dict, score_thres=default_params.get("score_thres"))
        # unstack gt info
        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
            unstacked_labels = unstack_batch(labels)
            gt_boxes_list = unstacked_labels["gt_boxes"]
            gt_labels_list = unstacked_labels["gt_labels"]
            # -1 due to label offset
            gt_labels_onehot_list = [tf.one_hot(tf.squeeze(tf.cast(gt_labels-label_offset, tf.int32), 1), num_classes)
                                     for gt_labels in gt_labels_list]
            reg_loss, cls_loss, box_weights, cls_weights = model.loss(predictions_dict, gt_boxes_list, gt_labels_onehot_list)
            losses = [reg_loss * default_params.get("box_loss_weight"), cls_loss]
            total_loss_dict = {"Loss/classification_loss": cls_loss, "Loss/localization_loss": reg_loss}
            # add regularization loss
            regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if regularization_loss:
                regularization_loss = tf.add_n(regularization_loss, name='regularization_loss')
                losses.append(regularization_loss)
                total_loss_dict["Loss/regularization_loss"] = regularization_loss
            total_loss = tf.add_n(losses, name='total_loss')
            total_loss_dict["Loss/total_loss"] = total_loss

            # optimizer
            if mode == tf.estimator.ModeKeys.TRAIN:
                lr = learning_rate_schedule(default_params.get("total_train_steps"))
                optimizer = tf.train.MomentumOptimizer(lr, momentum=default_params.get("momentum"))
                # batch norm need update_ops
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(total_loss, tf.train.get_global_step())
            else:
                train_op = None

        # predict mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {tf.saved_model.signature_constants.PREDICT_METHOD_NAME: detections}

        eval_metric_ops = {}
        # just for debugging
        logging_hook = [tf.train.LoggingTensorHook({"gt_labels": gt_labels_list[0], "gt_boxes": gt_boxes_list[0],
                                                    'norm_box_loss': reg_loss, 'norm_cls_loss': cls_loss,
                                                    "pred_box": predictions_dict["box_pred"],
                                                    "pred_cls": predictions_dict["cls_pred"]},
                                                   every_n_iter=50)]
        if mode == tf.estimator.ModeKeys.EVAL:
            logging_hook = [tf.train.LoggingTensorHook({"gt_labels": gt_labels_list[0], "gt_boxes": gt_boxes_list[0],
                                                        "detection_boxes": detections["detection_boxes"],
                                                        "detection_classes": detections["detection_classes"],
                                                        "scores": detections["detection_scores"],
                                                        "num_detections": detections["num_detections"]},
                                                       every_n_iter=50)]
            eval_dict = _result_dict_for_single_example(images[0:1], keys[0], detections,
                                                        gt_boxes_list[0], tf.reshape(gt_labels_list[0], [-1]))
            if run_config["label_map_path"] is None:
                raise RuntimeError("label map file must be defined first!")
            else:
                category_index = create_categories_from_labelmap(run_config["label_map_path"])
                coco_evaluator = CocoDetectionEvaluator(categories=category_index)
            eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
            eval_metric_ops["classification_loss"] = tf.metrics.mean(cls_loss)
            eval_metric_ops["localization_loss"] = tf.metrics.mean(reg_loss)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=detections,
                                          loss=total_loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops,
                                          training_hooks=logging_hook,
                                          export_outputs=export_outputs,
                                          evaluation_hooks=logging_hook)
    return model_fn


def create_train_and_eval_specs(train_input_fn,
                                eval_input_fn,
                                predict_fn,
                                train_steps):
    """
    Create a TrainSpec and EvalSpec
    """
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=train_steps)
    eval_spec_name = "0"
    exported_name = "{}_{}".format('Servo', eval_spec_name)
    exporter = tf.estimator.FinalExporter(name=exported_name, serving_input_receiver_fn=predict_fn)
    eval_spec = tf.estimator.EvalSpec(name=eval_spec_name, input_fn=eval_input_fn, steps=None, exporters=exporter)
    return train_spec, eval_spec

