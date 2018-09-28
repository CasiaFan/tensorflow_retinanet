import tensorflow as tf
from object_detection.tf_example_decoder import TfExampleDecoder
from object_detection import preprocessor


PARAMS = {"batch_size": 64,
          "image_size": (640, 640),
          "num_classes": 90,
          "max_instance": 100}


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
    with tf.control_dependencies([assert_count]):
        pad_len = max_count - num_instance
        padding = tf.ones(shape=(pad_len, dim)) * pad_value
        padded_inputs = tf.concat([inputs, padding], axis=0)
        return padded_inputs


def create_input_fn(file_pattern, is_training=True, params=PARAMS):
    """
    Create input reader for estimator model
    Returns `features` and `labels` tensor dictionaries for training or evaluation.

    Returns:
      A tf.data.Dataset that holds (features, labels) tuple.

      features: Dictionary of feature tensors.
        features["image"] is a [batch_size, H, W, C] float32 tensor with preprocessed images.
        features["true_image_size"] is a [batch_size, 2] int32 tensor representing true image shapes
      labels: Dictionary of groundtruth tensors.
        labels["gt_boxes"] is a [batch_size, num_boxes, 4] float32 tensor containing the NORMALIZED corners of the groundtruth boxes.
        labels["gt_classes"] is a [batch_size, num_boxes, num_classes] float32 one-hot tensor of classes.
    """
    def _preprocess(example):
        """
        example: a string tensor holding serialized tf example proto
        """
        data = example_decoder.decode(example)
        image = data["image"]
        boxes = data["groundtruth_boxes"]
        labels = data["groundtruth_classes"]
        true_image_width = data["image/width"]
        true_image_height = data["image/height"]
        # augmentation
        if is_training:
            image = preprocessor.random_adjust_brightness(image)
            image = preprocessor.random_adjust_saturation(image)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image, boxes = preprocessor.random_horizontal_flip(image, boxes)
            image, boxes, labels = preprocessor.random_crop_image(image, boxes, labels)
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # resize image
        image = tf.image.resize_images(image, size=params["image_size"])
        image = (image - 0.5) * 2
        boxes = pad_to_fixed_size(boxes, [params["max_instance"], 4], -1)
        labels = pad_to_fixed_size(labels, [params["max_instance"], 1], -1)
        image_shape = tf.stack([true_image_height, true_image_width], axis=0)
        return (image, boxes, labels, image_shape)

    def _parse_example(image, boxes, labels, image_shape):
        features, labels = {}, {}
        features["image"] = image
        features["true_image_shape"] = image_shape
        labels["gt_boxes"] = boxes,
        labels["gt_labels"] = labels

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
    dataset = dataset.prefetch(params["batch_size"])
    dataset = dataset.batch(params["batch_size"])
    # parse to dict
    dataset = dataset.map(_parse_example)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


def create_prediction_input_fn():
    """
    Create an input function for estimator model prediction
    """
    example_decode = TfExampleDecoder()
    example = tf.placeholder(dtype=tf.string, shape=[], name="tf_example")
    data = example_decode.decode(example)
    image = tf.image.convert_image_dtype(data["image"], dtype=tf.float32)
    image = (image - 0.5) * 2
    image = tf.expand_dims(image, 0)
    return tf.estimator.export.ServingInputReceiver(features={"image": image},
                                                    receiver_tensors={"serialized_example": example})


