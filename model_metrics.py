import tensorflow as tf
import numpy as np
from tqdm import tqdm
from absl import flags

FLAGS = flags.FLAGS


def evaluate_p_c_given_x(dataset, model, batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  keras_metrics = {
    'p(c|x) accuracy': tf.keras.metrics.BinaryAccuracy(),
    'p(c|x) log-likelihood': tf.keras.metrics.BinaryCrossentropy(),
    'p(c|x) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    concept_prob, _ = model.p_c_z_given_x(batch['data'])
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(batch['concepts'], concept_prob)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(c|x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_all_c_given_x(dataset, model, batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  keras_metrics = {
    'p(all c|x) accuracy': tf.keras.metrics.Mean(),
    'p(all c|x) log-likelihood': tf.keras.metrics.Mean()
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    c_samples, _, _, _ = model.sample_p_c_z_given_x(batch['data'], FLAGS.mc_samples_for_prediction, training=False)
    matches = tf.reduce_sum(tf.cast(tf.reduce_all(c_samples[None, :, :, :] == c_samples[:, None, :, :], axis=-1), dtype=tf.float32), axis=0)
    most_frequent = tf.math.argmax(matches, axis=0)
    c_pred = tf.gather(tf.transpose(c_samples, [1, 0, 2]), most_frequent, batch_dims=1)
    c = tf.cast(batch['concepts'], dtype=tf.float32)
    keras_metrics['p(all c|x) accuracy'].update_state(tf.cast(tf.reduce_all(c_pred == c, axis=-1), dtype=tf.float32))
    _, _, c_ll, _ = model.sample_p_c_z_given_x_masked(batch['data'], c, tf.ones_like(c, dtype=tf.bool), 1,
                                                      training=False)
    keras_metrics['p(all c|x) log-likelihood'].update_state(-tf.reduce_sum(c_ll[0, ...], axis=-1))

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(all c|x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_c_x(dataset,
                           model,
                           n_classes,
                           batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(y|c x) accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'p(y|c x) log-likelihood': tf.keras.metrics.CategoricalCrossentropy(),
    'p(y|c x) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    if model.type == 'joint' or model.type == 'sequential':
      c = tf.cast(batch['concepts'], tf.float32) * (model.c_95 - model.c_5) + model.c_5
    else:
      c = tf.cast(batch['concepts'], tf.float32)
    label_probs = model.p_y_given_x_c(batch['data'], c)
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(tf.one_hot(batch['label'], depth=n_classes), label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|c,x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_c(dataset,
                         model,
                         n_classes,
                         batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(y|c) accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'p(y|c) log-likelihood': tf.keras.metrics.CategoricalCrossentropy(),
    'p(y|c) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    if model.type == 'joint' or model.type == 'sequential':
      c = tf.cast(batch['concepts'], tf.float32) * (model.c_95 - model.c_5) + model.c_5
    else:
      c = tf.cast(batch['concepts'], tf.float32)
    label_probs = model.p_y_given_c(c)
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(tf.one_hot(batch['label'], depth=n_classes), label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|c) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_x(dataset,
                         model,
                         n_classes,
                         batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(y|x) accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'p(y|x) log-likelihood': tf.keras.metrics.CategoricalCrossentropy(),
    'p(y|x) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    label_probs = model.p_y_given_x(batch['data'])
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(tf.one_hot(batch['label'], depth=n_classes), label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_x_markovian(dataset,
                                   model,
                                   n_classes,
                                   batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(y|x) markovian accuracy': tf.keras.metrics.CategoricalAccuracy(),
    'p(y|x) markovian log-likelihood': tf.keras.metrics.CategoricalCrossentropy(),
    'p(y|x) markovian auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    label_probs = model.p_y_given_x_markovian(batch['data'])
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(tf.one_hot(batch['label'], depth=n_classes), label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|x) markovian metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y(dataset,
                 n_classes,
                 batch_size=512):
  dataset_size = tf.data.experimental.cardinality(dataset)
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  class_counts = tf.zeros((n_classes,), dtype=tf.float32)

  @tf.function
  def metrics_accumulator_fn(batch, class_counts):
    return class_counts + tf.reduce_sum(tf.one_hot(batch['label'], depth=n_classes), axis=0)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating H(y)'):
    class_counts = metrics_accumulator_fn(batch, class_counts)

  class_probs = tf.cast(class_counts, dtype=tf.float32) / tf.cast(dataset_size, dtype=tf.float32)
  numpy_metrics = {
    'H(y)': (-tf.reduce_sum(class_probs * tf.math.log(class_probs))).numpy()
  }
  return numpy_metrics


def evaluate_p_miss_given_x(dataset,
                            model,
                            batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(miss|x) accuracy': tf.keras.metrics.BinaryAccuracy(),
    'p(miss|x) log-likelihood': tf.keras.metrics.BinaryCrossentropy(),
    'p(miss|x) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    label_probs = model.p_miss_given_x(batch['data'])
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(batch['missing_concept'], label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(miss|x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_miss_c_x(dataset,
                                model,
                                batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    'p(y|miss c x) accuracy': tf.keras.metrics.BinaryAccuracy(),
    'p(y|miss c x) log-likelihood': tf.keras.metrics.BinaryCrossentropy(),
    'p(y|miss c x) auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  @tf.function
  def metrics_accumulator_fn(batch):
    label_probs = model.p_y_given_miss_c_x(batch['missing_concept'][:, None], batch['concepts'], batch['data'])
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(batch['label'], label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|miss,c,x) metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics


def evaluate_p_y_given_x_intervention(dataset,
                                      model,
                                      n_classes,
                                      intervention_mask,
                                      name,
                                      batch_size=512):
  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  keras_metrics = {
    name + ' intervention accuracy': tf.keras.metrics.CategoricalAccuracy(),
    name + ' intervention log-likelihood': tf.keras.metrics.CategoricalCrossentropy(),
    name + ' intervention auc': tf.keras.metrics.AUC(num_thresholds=500)
  }

  print(f'Needs adjustment {model.c_95} {model.c_5}')

  @tf.function
  def metrics_accumulator_fn(batch):
    if model.type == 'joint' or model.type == 'sequential':
      c = tf.cast(batch['concepts'], tf.float32) * (model.c_95 - model.c_5) + model.c_5
    else:
      c = tf.cast(batch['concepts'], tf.float32)
    label_probs = model.p_y_given_x_intervention(batch['data'], c, intervention_mask)
    for metric_name in keras_metrics:
      keras_metrics[metric_name].update_state(tf.one_hot(batch['label'], depth=n_classes), label_probs)

  for batch in tqdm(dataset,
                    total=int(tf.data.experimental.cardinality(dataset)),
                    desc='Calculating p(y|x) intervention metrics'):
    metrics_accumulator_fn(batch)

  numpy_metrics = {}
  for metric_name in keras_metrics:
    result = keras_metrics[metric_name].result().numpy()
    numpy_metrics[metric_name] = result
  return numpy_metrics
