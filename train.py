import tensorflow as tf
from model_config import config
from trainer import train

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags

flags.DEFINE_string('train_dir', 'model/train',
                    'Directory to save the checkpoints and training summaries.')

FLAGS = flags.FLAGS

def main(_):
    train_config = config.TRAIN
    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    train(train_config, FLAGS.train_dir, master, task,
          ps_tasks=ps_tasks,
          worker_job_name=worker_job_name,
          worker_replicas=worker_replicas,
          is_chief=is_chief)

if __name__ == '__main__':
  tf.app.run()