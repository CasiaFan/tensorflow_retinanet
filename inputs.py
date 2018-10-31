import tensorflow as tf
from object_detection.tf_example_decoder import TfExampleDecoder
from object_detection import preprocessor


def pad_to_fixed_size(inputs, output_shape, pad_value):
    """
    Pad inputs data to given output shape with given pad_value
    Args:
        output_shape: output shape of 2D tensor
    """
    max_count = output_shape[0]
    dim = output_shape[1]
    num_instance = tf.shape(inputs)[0]
    assert_count = tf.Assert(tf.less_equal(num_instance, max_count), [num_instance])
    inputs = tf.cast(inputs, tf.float32)
    with tf.control_dependencies([assert_count]):
        pad_len = max_count - num_instance
        padding = tf.cast(tf.ones(shape=(pad_len, dim)) * pad_value, tf.float32)
        padded_inputs = tf.concat([inputs, padding], axis=0)
        return padded_inputs


def create_input_fn(file_pattern, is_training, image_size, batch_size=1, num_max_instance=100, label_offset=1):
    """
    Create input reader for estimator model
    Returns `features` and `labels` tensor dictionaries for training or evaluation.

    Returns:
      A tf.data.Dataset that holds (features, labels) tuple.

      features: Dictionary of feature tensors.
        features["image"] is a [batch_size, H, W, C] float32 tensor with preprocessed images.
        features["true_image_size"] is a [batch_size, 2] int32 tensor representing true image shapes
                Note this will be used in clip_window of [0, 0, 1, 1] during post-processing
      labels: Dictionary of groundtruth tensors.
        labels["gt_boxes"] is a [batch_size, num_boxes, 4] shape float32 tensor containing the NORMALIZED corners of the groundtruth boxes.
        labels["gt_classes"] is a [batch_size, num_boxes, 1] shape tensor of classes.
    """
    def _input_fn():
        def _preprocess(example):
            """
            example: a string tensor holding serialized tf example proto
            """
            data = example_decoder.decode(example)
            image = data["image"]
            key = data["key"]
            boxes = data["groundtruth_boxes"]
            gt_classes = data["groundtruth_classes"]
            # label offset
            # gt_classes -= label_offset
            # augmentation
            if is_training:
                # image = preprocessor.random_adjust_brightness(image)
                # image = preprocessor.random_adjust_saturation(image)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                # image, boxes = preprocessor.random_horizontal_flip(image, boxes)
                # image, boxes, gt_classes = preprocessor.random_crop_image(image, boxes, gt_classes)
            else:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # resize image
            image = tf.image.resize_images(image, size=(image_size, image_size))
            image = (image - 0.5) * 2
            # num of gt boxes
            num_gt_boxes = tf.minimum(tf.shape(boxes)[0], num_max_instance)
            boxes = pad_to_fixed_size(boxes, [num_max_instance, 4], -1)
            gt_classes = pad_to_fixed_size(tf.expand_dims(gt_classes, 1), [num_max_instance, 1], -1)
            # gt_classes = tf.squeeze(gt_classes, 1)
            return (image, key, boxes, gt_classes, num_gt_boxes)

        def _parse_example(image, key, boxes, gt_classes, num_gt_boxes):
            features, labels = {}, {}
            features["image"] = image
            features["key"] = key
            labels["gt_boxes"] = boxes,
            labels["gt_labels"] = gt_classes
            labels["num_gt_boxes"] = num_gt_boxes
            return features, labels

        example_decoder = TfExampleDecoder()
        # get tfrecord file
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        if is_training:
            dataset = dataset.repeat()
        # prefetch data from tfrecord file
        prefetch_func = lambda x: tf.data.TFRecordDataset(x).prefetch(1)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            prefetch_func, block_length=2, cycle_length=32, sloppy=is_training))
        # shuffle
        if is_training:
            dataset = dataset.shuffle(64)
        dataset = dataset.map(_preprocess, num_parallel_calls=64)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # if is_training:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # parse to dict
        dataset = dataset.map(_parse_example)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
    return _input_fn


def create_prediction_input_fn():
    """
    Create an input function for estimator model prediction
    """
    def _pred_input_fn():
        pass
        # example_decode = TfExampleDecoder()
        # example = tf.placeholder(dtype=tf.string, shape=[], name="tf_example")
        # data = example_decode.decode(example)
        # image = tf.image.convert_image_dtype(data["image"], dtype=tf.float32)
        # image = (image - 0.5) * 2
        # image = tf.expand_dims(image, 0)
        # return tf.estimator.export.ServingInputReceiver(features={"image": image},
        #                                                 receiver_tensors={"serialized_example": example})
    return _pred_input_fn()

