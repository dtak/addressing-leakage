from absl import app
from absl import flags
import tf_dataset as tf_cub
import concept_bottleneck_model
import model_metrics
import numpy as np
import tensorflow as tf
import os

FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset_name', 'cub', ['cub'], 'Datasets name.')
flags.DEFINE_integer('train_epochs', 100, 'Number of training epochs.')
flags.DEFINE_integer('amortization_epochs', 10, 'Number of amortization epochs.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('evaluation_batch_size', 16, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')
flags.DEFINE_integer('random_seed', 0, 'Random seed.')
flags.DEFINE_float('lamb', 1., 'Lambda for joint training.')
flags.DEFINE_enum('model_type', 'hard', ['hard', 'independent', 'sequential', 'joint'], 'Soft training.')
flags.DEFINE_integer('latent_dims', 0, 'Number of latent dimensions.')
flags.DEFINE_enum('eval_set', 'test', ['all', 'training', 'test'], 'Datasets to evaluate on.')
flags.DEFINE_boolean('overwrite_metrics', True, 'Overwrite the metrics save file.')
flags.DEFINE_integer('n_groups', -1, 'Number of concept groups to use.')
flags.DEFINE_string('save_metrics',
                    'output_metrics.csv',
                    'Save metrics in this file.')
flags.DEFINE_integer('pretrain_autoregressive', 20, 'Number of pretraining epochs for autoregressive.')

def main(_):
  train_set = tf_cub.get_dataset(train=True, augmented=True)
  train_set_unaugmented = tf_cub.get_dataset(train=True, augmented=False)
  test_set = tf_cub.get_dataset(train=False, augmented=False)
  n_concepts = train_set.element_spec['concepts'].shape[0]
  n_classes = tf_cub.num_classes
  concept_groups = tf_cub.get_concept_groups()
  concept_groups = concept_groups[:, ::-1]

  if FLAGS.n_groups != -1:
    n_groups = concept_groups.shape[0]
    keep_mask = np.sum(concept_groups[:FLAGS.n_groups, :], axis=0)
    n_keep = int(np.sum(keep_mask))

    def mask_concepts(input):
      input['concepts'] = input['concepts'][:n_keep]
      return input
    n_concepts = n_keep
    concept_groups = concept_groups[:FLAGS.n_groups, :n_keep]
    train_set = train_set.map(mask_concepts)
    train_set_unaugmented = train_set_unaugmented.map(mask_concepts)
    test_set = test_set.map(mask_concepts)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)
  # tf.debugging.enable_check_numerics(
  #   stack_height_limit=30, path_length_limit=50
  # )

  
  model = concept_bottleneck_model.ConceptBottleneckModel(
        n_concepts,
        FLAGS.latent_dims,
        n_classes,
        use_inceptionv3=True,
        type=FLAGS.model_type)
  model.train('sgd', train_set, FLAGS.train_epochs, FLAGS.batch_size, FLAGS.learning_rate,
                  amortization_epochs=FLAGS.amortization_epochs,
                  soft_gradient=FLAGS.model_type, burn_in=False,
                  label_lr_multiplier=1.0, pretrain_autoregressive=FLAGS.pretrain_autoregressive)
   
  if FLAGS.eval_set == 'all':
    eval_sets = {
      'train': train_set_unaugmented,
      'test': test_set
    }
  elif FLAGS.eval_set == 'train':
    eval_sets = {
      'train': train_set_unaugmented
    }
  elif FLAGS.eval_set == 'test':
    eval_sets = {
      'test': test_set
    }

  save_metrics = {}
  for set_name in eval_sets:
    save_metrics[set_name] = {}

  for set_name in eval_sets:

    if model.type == 'joint' or model.type == 'sequential':
      model.compute_c_percentile(train_set, batch_size=FLAGS.batch_size)

    metrics = model_metrics.evaluate_p_all_c_given_x(eval_sets[set_name], model, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_c_given_x(eval_sets[set_name], model, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_y_given_c(eval_sets[set_name], model, n_classes, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_y_given_x(eval_sets[set_name], model, n_classes, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_y_given_c_x(eval_sets[set_name], model, n_classes, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_y_given_x_markovian(eval_sets[set_name], model, n_classes, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)

    metrics = model_metrics.evaluate_p_y(eval_sets[set_name], n_classes, batch_size=FLAGS.evaluation_batch_size)
    for metric_name in metrics:
      print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
    save_metrics[set_name].update(metrics)


    intervention_mask = tf.zeros((n_concepts,), dtype=np.float32)
    for gi in range(concept_groups.shape[0] + 1):
      metrics = model_metrics.evaluate_p_y_given_x_intervention(eval_sets[set_name], model, n_classes, intervention_mask, name=str(gi), batch_size=FLAGS.evaluation_batch_size)
      for metric_name in metrics:
        print('  ' + metric_name + ': %0.3f' % metrics[metric_name])
      save_metrics[set_name].update(metrics)
      if gi == concept_groups.shape[0]:
        break
      else:
        intervention_mask = intervention_mask + concept_groups[gi, :]

  if FLAGS.save_metrics is not None:
    header = ['Data split']
    data = []
    for set_name in save_metrics:
      save_metrics[set_name]['seed'] = FLAGS.random_seed
      for metric_name in save_metrics[set_name]:
        if metric_name not in header:
          header.append(metric_name)

    for set_name in save_metrics:
      line = [set_name] + ([None] * (len(header) - 1))

      for metric_name in save_metrics[set_name]:
        line[header.index(metric_name)] = save_metrics[set_name][metric_name]
      data.append(line)
    if FLAGS.overwrite_metrics or not os.path.exists(FLAGS.save_metrics):
      np.savetxt(FLAGS.save_metrics, data, header=','.join(header), delimiter=',', fmt='%s')
    else:
      with open(FLAGS.save_metrics, "ab") as f:
        np.savetxt(f, data, delimiter=',', fmt='%s')


if __name__ == '__main__':
  app.run(main)
