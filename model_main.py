import tensorflow as tf
from absl import flags
from model import create_model_fn, create_train_and_eval_specs
from inputs import create_prediction_input_fn, create_input_fn


flags.DEFINE_string('model_dir', None, 'Path to output model directory')
flags.DEFINE_integer('num_train_steps', 200000, 'Number of train steps')
flags.DEFINE_string('train_file_pattern', None, 'Pattern of tfrecord file for training')
flags.DEFINE_string('eval_file_pattern', None, 'Pattern of tfrecord file for evaluation')
flags.DEFINE_integer('image_size', 224, 'Input image shape')
flags.DEFINE_integer('num_classes', 1, 'Number of total classes')
flags.DEFINE_integer('batch_size', 2, 'Number of batch size for training')
flags.DEFINE_string('finetune_ckpt', None, 'Path to finetune checkpoint file')
flags.DEFINE_string('label_map_path', None, 'Path to label map file in tensorflow object detection api format')
FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('label_map_path')
    flags.mark_flag_as_required('train_file_pattern')
    flags.mark_flag_as_required('eval_file_pattern')

    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
    run_config = {"label_map_path": FLAGS.label_map_path,
                  "num_classes": FLAGS.num_classes}
    if FLAGS.finetune_ckpt:
        run_config["finetune_ckpt"] = FLAGS.finetune_ckpt
    model_fn = create_model_fn(run_config)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
    train_input_fn = create_input_fn(FLAGS.train_file_pattern, True, FLAGS.image_size, FLAGS.batch_size)
    eval_input_fn = create_input_fn(FLAGS.eval_file_pattern, False, FLAGS.image_size)
    prediction_fn = create_prediction_input_fn()
    train_spec, eval_spec = create_train_and_eval_specs(train_input_fn, eval_input_fn, prediction_fn, FLAGS.num_train_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)