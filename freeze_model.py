"""
Freeze estimator models into frozen graph model file (.pb)
"""

import tensorflow as tf
import os
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib


def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=input_shape, name='image_tensor')
    return input_tensor, input_tensor


def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
    """Adds output nodes for detection boxes and scores, with following items:
        num_detections, detection_boxes, detection_scores, detection_classes
    Args:
        postprocessed_tensors: a dictionary containing the following fields
            'detection_boxes': [batch, max_detections, 4]
            'detection_scores': [batch, max_detections]
            'detection_classes': [batch, max_detections]
        output_collection_name: Name of collection to add output tensors to.

    Returns:
        A tensor dict containing the added output tensor nodes.
    """
    label_id_offset = 1
    boxes = postprocessed_tensors.get("detection_boxes")
    scores = postprocessed_tensors.get("detection_scores")
    classes = postprocessed_tensors.get("detection_classes") + label_id_offset
    num_detections = postprocessed_tensors.get("num_detections")
    outputs = {}
    outputs["detection_boxes"] = tf.identity(boxes, name="detection_boxes")
    outputs["detection_scores"] = tf.identity(scores, name="detection_scores")
    outputs["detection_classes"] = tf.identity(classes, name="detection_classes")
    outputs["num_detections"] = tf.identity(num_detections, name="num_detections")
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])
    return outputs


def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
    """Writes SavedModel to disk for tensorflow serving.
    Args:
        saved_model_path: Path to write SavedModel.
        frozen_graph_def: tf.GraphDef holding frozen graph.
        inputs: The input placeholder tensor.
        outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)
            detection_signature = (
              tf.saved_model.signature_def_utils.build_signature_def(
                  inputs=tensor_info_inputs,
                  outputs=tensor_info_outputs,
                  method_name=signature_constants.PREDICT_METHOD_NAME))
            builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.tag_constants.SERVING],
              signature_def_map={
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      detection_signature,
              },
            )
            builder.save()


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    """Writes the graph and the checkpoint into disk."""
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with session.Session() as sess:
            saver = saver_lib.Saver(saver_def=input_saver_def,
                                    save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, detection_model, input_size, output_collection_name):
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs = detection_model.preprocess(inputs, input_size)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)
    return _add_output_tensor_nodes(postprocessed_tensors,
                                  output_collection_name)


def _build_detection_graph(input_type, detection_model, input_shape,
                           output_collection_name, graph_hook_fn):
  """Build the detection graph."""
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  placeholder_args = {}
  if input_shape is not None:
    if input_type != 'image_tensor':
      raise ValueError('Can only specify input shape for `image_tensor` '
                       'inputs.')
    placeholder_args['input_shape'] = input_shape
  placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](
      **placeholder_args)
  outputs = _get_outputs_from_inputs(
      input_tensors=input_tensors,
      detection_model=detection_model,
      output_collection_name=output_collection_name)

  # Add global step to the graph.
  slim.get_or_create_global_step()

  if graph_hook_fn: graph_hook_fn()

  return outputs, placeholder_tensor


def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None,
                            write_inference_graph=False):
  """Export helper."""
  tf.gfile.MakeDirs(output_directory)
  frozen_graph_path = os.path.join(output_directory,
                                   'frozen_inference_graph.pb')
  saved_model_path = os.path.join(output_directory, 'saved_model')
  model_path = os.path.join(output_directory, 'model.ckpt')

  outputs, placeholder_tensor = _build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn)

  profile_inference_graph(tf.get_default_graph())
  saver_kwargs = {}
  if use_moving_averages:
    # This check is to be compatible with both version of SaverDef.
    if os.path.isfile(trained_checkpoint_prefix):
      saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
      temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
    else:
      temp_checkpoint_prefix = tempfile.mkdtemp()
    replace_variable_values_with_moving_averages(
        tf.get_default_graph(), trained_checkpoint_prefix,
        temp_checkpoint_prefix)
    checkpoint_to_use = temp_checkpoint_prefix
  else:
    checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()

  write_graph_and_checkpoint(
      inference_graph_def=tf.get_default_graph().as_graph_def(),
      model_path=model_path,
      input_saver_def=input_saver_def,
      trained_checkpoint_prefix=checkpoint_to_use)
  if write_inference_graph:
    inference_graph_def = tf.get_default_graph().as_graph_def()
    inference_graph_path = os.path.join(output_directory,
                                        'inference_graph.pbtxt')
    for node in inference_graph_def.node:
      node.device = ''
    with gfile.GFile(inference_graph_path, 'wb') as f:
      f.write(str(inference_graph_def))

  if additional_output_tensor_names is not None:
    output_node_names = ','.join(outputs.keys()+additional_output_tensor_names)
  else:
    output_node_names = ','.join(outputs.keys())

  frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=frozen_graph_path,
      clear_devices=True,
      initializer_nodes='')

  write_saved_model(saved_model_path, frozen_graph_def,
                    placeholder_tensor, outputs)


def export_inference_graph(input_type,
                           pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None,
                           write_inference_graph=False):
  """Exports inference graph for the model specified in the pipeline config.

  Args:
    input_type: Type of input for the graph. Can be one of ['image_tensor',
      'encoded_image_string_tensor', 'tf_example'].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_prefix: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
    input_shape: Sets a fixed shape for an `image_tensor` input. If not
      specified, will default to [None, None, None, 3].
    output_collection_name: Name of collection to add output tensors to.
      If None, does not add output tensors to a collection.
    additional_output_tensor_names: list of additional output
      tensors to include in the frozen graph.
    write_inference_graph: If true, writes inference graph to disk.
  """
  detection_model = model_builder.build(pipeline_config.model,
                                        is_training=False)
  graph_rewriter_fn = None
  if pipeline_config.HasField('graph_rewriter'):
    graph_rewriter_config = pipeline_config.graph_rewriter
    graph_rewriter_fn = graph_rewriter_builder.build(graph_rewriter_config,
                                                     is_training=False)
  _export_inference_graph(
      input_type,
      detection_model,
      pipeline_config.eval_config.use_moving_averages,
      trained_checkpoint_prefix,
      output_directory,
      additional_output_tensor_names,
      input_shape,
      output_collection_name,
      graph_hook_fn=graph_rewriter_fn,
      write_inference_graph=write_inference_graph)
  pipeline_config.eval_config.use_moving_averages = False
  config_util.save_pipeline_config(pipeline_config, output_directory)


def profile_inference_graph(graph):
  """Profiles the inference graph.

  Prints model parameters and computation FLOPs given an inference graph.
  BatchNorms are excluded from the parameter count due to the fact that
  BatchNorms are usually folded. BatchNorm, Initializer, Regularizer
  and BiasAdd are not considered in FLOP count.

  Args:
    graph: the inference graph.
  """
  tfprof_vars_option = (
      tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  tfprof_flops_option = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS

  # Batchnorm is usually folded during inference.
  tfprof_vars_option['trim_name_regexes'] = ['.*BatchNorm.*']
  # Initializer and Regularizer are only used in training.
  tfprof_flops_option['trim_name_regexes'] = [
      '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
  ]

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      graph,
      tfprof_options=tfprof_vars_option)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      graph,
      tfprof_options=tfprof_flops_option)
