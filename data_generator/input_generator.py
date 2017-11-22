import collections
import tensorflow as tf
from utils.preprocess import preprocess

slim_example_decoder = tf.contrib.slim.tfexample_decoder
parallel_reader = tf.contrib.slim.parallel_reader


def input_queue_generator(input_reader_config):
    """
    input_reader_config: a dict configure the input reader
    input path: tf record file path
    num_epochs: epoch of training input images, default is None (loop infinitely)
    num_readers: integer, the num of Readers to create
    shuffle: boolean, use RandomShuffleQueue to shuffle files and records
    capacity: integer, capacity of queue
    """
    _, string_tensor = parallel_reader.parallel_read(
        input_reader_config.input_path,
        reader_class=tf.TFRecordReader,
        num_epochs=(input_reader_config.num_epochs
                    if input_reader_config.num_epochs else None),
        num_readers=input_reader_config.num_readers,
        shuffle=input_reader_config.shuffle,
        dtypes=[tf.string, tf.string],
        capacity=input_reader_config.queue_capacity)

    tensor_dict = TfExampleDecoder().decode(string_tensor)
    # input image (convert to float32 type)
    tensor_dict['image'] = tf.to_float(tf.expand_dims(tensor_dict['image'], 0))
    if input_reader_config.data_augmentation_ops:
        tensor_dict = preprocess(tensor_dict, input_reader_config.data_augmentation_ops)
    input_queue = BatchQueue(
        tensor_dict,
        batch_size=input_reader_config.batch_size,
        batch_queue_capacity=input_reader_config.batch_queue_capacity,
        num_batch_queue_threads=input_reader_config.num_batch_queue_threads,
        prefetch_queue_capacity=input_reader_config.prefetch_queue_capacity
    )
    return input_queue


class TfExampleDecoder():
  """Tensorflow Example proto decoder."""

  def __init__(self):
    """Constructor sets keys_to_features and items_to_handlers."""
    self.keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        # Object boxes and classes.
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/difficult': tf.VarLenFeature(tf.int64),
    }
    self.items_to_handlers = {
        'image': slim_example_decoder.Image(
            image_key='image/encoded', format_key='image/format', channels=3),
        'source_id': (
            slim_example_decoder.Tensor('image/source_id')),
        'key': (
            slim_example_decoder.Tensor('image/key/sha256')),
        'filename': (
            slim_example_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        'groundtruth_boxes': (
            slim_example_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')),
        'groundtruth_classes': (
            slim_example_decoder.Tensor('image/object/class/label')),
        'groundtruth_difficult': (
            slim_example_decoder.Tensor('image/object/difficult')),
    }

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputDataFields.source_id - string tensor containing original
        image id.
      fields.InputDataFields.key - string tensor with unique sha256 hash key.
      fields.InputDataFields.filename - string tensor with original dataset
        filename.
      fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
    """

    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    tensor_dict['image'].set_shape([None, None, 3])
    return tensor_dict


def _prefetch(tensor_dict, capacity):
  """Creates a prefetch queue for tensors.

  Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a
  dequeue op that evaluates to a tensor_dict. This function is useful in
  prefetching preprocessed tensors so that the data is readily available for
  consumers.

  Example input pipeline when you don't need batching:
  ----------------------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)
  tensor_dict = prefetch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  ----------------------------------------------------

  For input pipelines with batching, refer to core/batcher.py

  Args:
    tensor_dict: a dictionary of tensors to prefetch.
    capacity: the size of the prefetch queue.

  Returns:
    a FIFO prefetcher queue
  """
  names = list(tensor_dict.keys())
  dtypes = [t.dtype for t in tensor_dict.values()]
  shapes = [t.get_shape() for t in tensor_dict.values()]
  prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes,
                                       shapes=shapes,
                                       names=names,
                                       name='prefetch_queue')
  enqueue_op = prefetch_queue.enqueue(tensor_dict)
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))
  tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name,
                                                      capacity),
                    tf.to_float(prefetch_queue.size()) * (1. / capacity))
  return prefetch_queue


rt_shape_str = '_runtime_shapes'
class BatchQueue(object):
  """BatchQueue class.

  This class creates a batch queue to asynchronously enqueue tensors_dict.
  It also adds a FIFO prefetcher so that the batches are readily available
  for the consumers.  Dequeue ops for a BatchQueue object can be created via
  the Dequeue method which evaluates to a batch of tensor_dict.

  Example input pipeline with batching:
  ------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  batch_queue = batcher.BatchQueue(tensor_dict,
                                   batch_size=32,
                                   batch_queue_capacity=2000,
                                   num_batch_queue_threads=8,
                                   prefetch_queue_capacity=20)
  tensor_dict = batch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  -----------------------------------

  Notes:
  -----
  This class batches tensors of unequal sizes by zero padding and unpadding
  them after generating a batch. This can be computationally expensive when
  batching tensors (such as images) that are of vastly different sizes. So it is
  recommended that the shapes of such tensors be fully defined in tensor_dict
  while other lightweight tensors such as bounding box corners and class labels
  can be of varying sizes. Use either crop or resize operations to fully define
  the shape of an image in tensor_dict.

  It is also recommended to perform any preprocessing operations on tensors
  before passing to BatchQueue and subsequently calling the Dequeue method.

  Another caveat is that this class does not read the last batch if it is not
  full. The current implementation makes it hard to support that use case. So,
  for evaluation, when it is critical to run all the examples through your
  network use the input pipeline example mentioned in core/prefetcher.py.
  """

  def __init__(self, tensor_dict, batch_size, batch_queue_capacity,
               num_batch_queue_threads, prefetch_queue_capacity):
    """Constructs a batch queue holding tensor_dict.

    Args:
      tensor_dict: dictionary of tensors to batch.
      batch_size: batch size.
      batch_queue_capacity: max capacity of the queue from which the tensors are
        batched.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: max capacity of the queue used to prefetch
        assembled batches.
    """
    # Remember static shapes to set shapes of batched tensors.
    static_shapes = collections.OrderedDict(
        {key: tensor.get_shape() for key, tensor in tensor_dict.items()})
    # Remember runtime shapes to unpad tensors after batching.
    runtime_shapes = collections.OrderedDict(
        {(key + rt_shape_str): tf.shape(tensor)
         for key, tensor in tensor_dict.items()})

    all_tensors = tensor_dict
    all_tensors.update(runtime_shapes)
    batched_tensors = tf.train.batch(
        all_tensors,
        capacity=batch_queue_capacity,
        batch_size=batch_size,
        dynamic_pad=True,
        num_threads=num_batch_queue_threads)

    self._queue = _prefetch(batched_tensors, prefetch_queue_capacity)
    self._static_shapes = static_shapes
    self._batch_size = batch_size

  def dequeue(self):
    """Dequeues a batch of tensor_dict from the BatchQueue.

    TODO: use allow_smaller_final_batch to allow running over the whole eval set

    Returns:
      A list of tensor_dicts of the requested batch_size.
    """
    batched_tensors = self._queue.dequeue()
    # Separate input tensors from tensors containing their runtime shapes.
    tensors = {}
    shapes = {}
    for key, batched_tensor in batched_tensors.items():
      unbatched_tensor_list = tf.unstack(batched_tensor)
      for i, unbatched_tensor in enumerate(unbatched_tensor_list):
        if rt_shape_str in key:
          shapes[(key[:-len(rt_shape_str)], i)] = unbatched_tensor
        else:
          tensors[(key, i)] = unbatched_tensor

    # Undo that padding using shapes and create a list of size `batch_size` that
    # contains tensor dictionaries.
    tensor_dict_list = []
    batch_size = self._batch_size
    for batch_id in range(batch_size):
      tensor_dict = {}
      for key in self._static_shapes:
        tensor_dict[key] = tf.slice(tensors[(key, batch_id)],
                                    tf.zeros_like(shapes[(key, batch_id)]),
                                    shapes[(key, batch_id)])
        tensor_dict[key].set_shape(self._static_shapes[key])
      tensor_dict_list.append(tensor_dict)

    return tensor_dict_list