import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import itertools
from tqdm import tqdm
from absl import flags
import time

tfd = tfp.distributions
FLAGS = flags.FLAGS

flags.DEFINE_float('epsilon', 1e-6, 'Epsilon tolerance.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate.')
flags.DEFINE_float('label_dropout', 0.0, 'Deprecated.')
flags.DEFINE_float('inception_weight_decay', 8e-4, 'Weight decay for InceptionV3.')
flags.DEFINE_integer('mc_samples_for_training', 10,
                     'Number of MC samples for prediction when the dimensions are too high.')
flags.DEFINE_integer('mc_samples_for_prediction', 200,
                     'Number of MC samples for prediction when the dimensions are too high.')
flags.DEFINE_integer('intermediate_size', 50,
                     'Size of intermediate layer for autoregressive predictions.')
flags.DEFINE_integer('label_multilayer', 200,
                     'Multilayer label predictor.')
flags.DEFINE_integer('autoregressive_multilayer', 50,
                     'Multilayer autoregressive predictor.')
flags.DEFINE_boolean('autoregressive', True,
                     'Autoregressive concept predictors.')


@tf.custom_gradient
def reinforce_mean(outcomes, log_probs):
  # First axis assumed to be the sample dimension
  # Outcomes last axis is predictions
  tf.ensure_shape(log_probs, outcomes.shape[:-1])

  def grad(upstream):
    return upstream * tf.ones_like(outcomes) / outcomes.shape[0], tf.reduce_sum(upstream * outcomes, axis=-1) / \
           outcomes.shape[0]

  return tf.reduce_mean(outcomes, axis=0), grad

def logit(x):
  """ Computes the logit function, i.e. the logistic sigmoid inverse. """
  return - tf.math.log(1. / (x + FLAGS.epsilon) - 1. + 2. * FLAGS.epsilon)

def stabilize_probability(prob):
  return FLAGS.epsilon + (1. - 2. * FLAGS.epsilon) * prob


class ConceptBottleneckModel(tf.keras.Model):
  def __init__(self, n_concepts, n_latent_concepts, n_label_classes, n_features=None, use_inceptionv3=False,
               type='hard'):
    super(ConceptBottleneckModel, self).__init__()
    self.n_concepts = n_concepts
    self.n_latent_concepts = n_latent_concepts
    self.n_label_classes = n_label_classes
    self.n_features = n_features
    self.use_inceptionv3 = use_inceptionv3
    self.soft = (type != 'hard')
    self.type = type
    if self.n_features is not None:
      assert not self.use_inceptionv3
      self.x_shape = [self.n_features]
      if FLAGS.autoregressive:
        inputs = tf.keras.Input(shape=self.x_shape)
        self.intermediate_predictor = tf.keras.Sequential()
        self.intermediate_predictor.add(tf.keras.Input(shape=(n_features,)))
        self.intermediate_predictor.add(tf.keras.layers.Dense(FLAGS.intermediate_size,
                                                              activation='linear'))

        self.autoregressive_predictors = []
        for i in range(self.n_concepts + self.n_latent_concepts):
          predictor = tf.keras.Sequential()
          predictor.add(tf.keras.Input(shape=i + FLAGS.intermediate_size))
          if FLAGS.autoregressive_multilayer != -1:
            predictor.add(tf.keras.layers.Dense(FLAGS.autoregressive_multilayer, activation='relu'))
          predictor.add(tf.keras.layers.Dense(1, activation='sigmoid'))
          self.autoregressive_predictors.append(predictor)
      else:
        self.concept_predictor = tf.keras.Sequential()
        self.concept_predictor.add(tf.keras.Input(shape=(n_features,)))
        self.concept_predictor.add(tf.keras.layers.Dense(100,
                                                         activation='relu'))
        self.concept_predictor.add(tf.keras.layers.Dense(self.n_concepts + n_latent_concepts,
                                                         activation='sigmoid'))
    else:
      assert self.use_inceptionv3
      img_shape = [299, 299, 3]
      self.x_shape = img_shape
      base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=img_shape,
                                                                  include_top=False,
                                                                  weights='imagenet')
      global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
      intermediate_layer = tf.keras.layers.Dense(FLAGS.intermediate_size, activation='linear')
      prediction_layer = tf.keras.layers.Dense(self.n_concepts + self.n_latent_concepts, activation='sigmoid')
      inputs = tf.keras.Input(shape=img_shape)
      if FLAGS.autoregressive:
        x = base_model(inputs)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(FLAGS.dropout)(x)
        x = intermediate_layer(x)
        self.intermediate_predictor = tf.keras.Model(inputs, x)

        self.autoregressive_predictors = []
        for i in range(self.n_concepts + self.n_latent_concepts):
          predictor = tf.keras.Sequential()
          predictor.add(tf.keras.Input(shape=i + FLAGS.intermediate_size))
          if FLAGS.autoregressive_multilayer != -1:
            predictor.add(tf.keras.layers.Dense(FLAGS.autoregressive_multilayer, activation='relu'))
          predictor.add(tf.keras.layers.Dense(1, activation='sigmoid'))
          self.autoregressive_predictors.append(predictor)
      else:
        x = base_model(inputs)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(FLAGS.dropout)(x)
        x = prediction_layer(x)
        self.concept_predictor = tf.keras.Model(inputs, x)

    if FLAGS.autoregressive:
      self.autoregressive_amortizers = []
      for i in range(self.n_latent_concepts):
        predictor = tf.keras.Sequential()
        predictor.add(tf.keras.Input(shape=i + self.n_concepts))
        if FLAGS.autoregressive_multilayer:
          predictor.add(tf.keras.layers.Dense(20, activation='relu'))
        predictor.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.autoregressive_amortizers.append(predictor)
    else:
      self.amortization_predictor = tf.keras.Sequential()
      self.amortization_predictor.add(tf.keras.Input(shape=(self.n_concepts,)))
      self.amortization_predictor.add(tf.keras.layers.Dense(100,
                                                            activation='relu'))
      self.amortization_predictor.add(tf.keras.layers.Dense(self.n_latent_concepts,
                                                            activation='sigmoid'))

    self.label_predictor = tf.keras.Sequential()
    self.label_predictor.add(tf.keras.Input(shape=(self.n_concepts + self.n_latent_concepts,)))
    if FLAGS.label_multilayer != -1:
      self.label_predictor.add(tf.keras.layers.Dense(FLAGS.label_multilayer,
                                                     activation='relu'))
      self.label_predictor.add(tf.keras.layers.Dropout(FLAGS.label_dropout))
    self.label_predictor.add(tf.keras.layers.Dense(self.n_label_classes,
                                                   activation='linear'))
    self.label_predictor.add(tf.keras.layers.Softmax())
    self.weight_prior = tfd.Normal(loc=0., scale=5.)
    self.input_mean = 0.
    self.input_std = 1.
    self.c_5 = 0.05
    self.c_95 = 0.95

  def get_prior_log_likelihood(self):
    if self.use_inceptionv3:
      if FLAGS.autoregressive:
        return FLAGS.inception_weight_decay * (tf.add_n(
          [tf.reduce_sum(w * w) for w in self.intermediate_predictor.trainable_weights]) +
                                               tf.add_n([tf.add_n([tf.reduce_sum(w * w) for w in a.trainable_weights])
                                                         for a in self.autoregressive_predictors]))
      else:
        return FLAGS.inception_weight_decay * tf.add_n(
          [tf.reduce_sum(w * w) for w in self.concept_predictor.trainable_weights])
    else:
      if FLAGS.autoregressive:
        return (tf.add_n(
          [tf.reduce_sum(self.weight_prior.log_prob(w)) for w in self.intermediate_predictor.trainable_weights]) +
                tf.add_n([tf.add_n([tf.reduce_sum(self.weight_prior.log_prob(w)) for w in a.trainable_weights])
                          for a in self.autoregressive_predictors]) +
                tf.add_n([tf.reduce_sum(self.weight_prior.log_prob(w)) for w in self.label_predictor.trainable_weights]))
      else:
        return tf.add_n(
          [tf.reduce_sum(self.weight_prior.log_prob(w)) for w in self.concept_predictor.trainable_weights] +
          [tf.reduce_sum(self.weight_prior.log_prob(w)) for w in self.label_predictor.trainable_weights])

  def normalize_inputs(self, dataset, batch_size=512):
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    sum = tf.zeros((dataset.element_spec['data'].shape[1],))
    squared_sum = tf.zeros((dataset.element_spec['data'].shape[1],))
    i = 0
    for inputs in tqdm(dataset, desc='Normalizing inputs'):
      sum += tf.reduce_sum(inputs['data'], axis=0)
      squared_sum += tf.reduce_sum(tf.math.pow(inputs['data'], 2), axis=0)
      i += inputs['data'].shape[0]
    self.input_mean = (sum / i).numpy()
    self.input_std = tf.sqrt(((squared_sum / i) - tf.math.pow(sum / i, 2))).numpy()

  def compute_c_percentile(self, dataset, batch_size=512):
    data_size = tf.data.experimental.cardinality(dataset)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    c_s = np.zeros((data_size, self.n_concepts), dtype=np.float32)
    true_c_s = np.zeros((data_size, self.n_concepts), dtype=np.float32)

    @tf.function
    def percentile_accumulator_fn(batch):
      concept_prob, _ = self.p_c_z_given_x(batch['data'])
      return concept_prob

    i = 0
    for batch in tqdm(dataset,
                      total=int(tf.data.experimental.cardinality(dataset)),
                      desc='Computing c percentile'):
      bs = batch['data'].shape[0]
      c_s[i:i + bs, :] = percentile_accumulator_fn(batch).numpy()
      true_c_s[i:i + bs, :] = batch['concepts']
      i += bs

    self.c_5 = np.zeros((self.n_concepts,), dtype=np.float32)
    for c_i in range(self.n_concepts):
      negative_probs = []
      for j in range(i):
        if true_c_s[j, c_i] == 0.:
          negative_probs.append(c_s[j, c_i])
      negative_probs.sort()
      self.c_5[c_i] = negative_probs[int(len(negative_probs) * 0.05)]

    self.c_95 = np.zeros((self.n_concepts,), dtype=np.float32)
    for c_i in range(self.n_concepts):
      positive_probs = []
      for j in range(i):
        if true_c_s[j, c_i] == 1.:
          positive_probs.append(c_s[j, c_i])
      positive_probs.sort()
      self.c_95[c_i] = positive_probs[int(len(positive_probs) * 0.95)]

  def call(self, x):
    return self.p_y_given_x(x)

  def predict_model(self, model, input, input_shape, training=False):
    batch_input_shape = input.shape
    batch_shape = batch_input_shape[:-len(list(input_shape))]
    input = tf.reshape(input, [np.prod(batch_shape, dtype=np.int32)] + input_shape)
    result = model(input, training=training)
    return tf.reshape(result, batch_shape + [result.shape[-1]])

  def sample_p_c_z_given_x_masked(self, x, c, c_mask, n_samples, training=False, pretrain=False):
    if not FLAGS.autoregressive:
      concept_and_latent_probs = self.predict_model(self.concept_predictor, self.get_normalized_inputs(x), self.x_shape,
                                                    training=training)
      concept_probs = concept_and_latent_probs[..., :self.n_concepts]
      latent_probs = concept_and_latent_probs[..., self.n_concepts:]
      concept_samples = tf.where(c_mask,
                                 c,
                                 tf.where(tf.random.uniform([n_samples] + concept_probs.shape) < concept_probs,
                                          1.,
                                          0.))
      latent_samples = tf.where(tf.random.uniform([n_samples] + latent_probs.shape) < latent_probs,
                                1.,
                                0.)
      concept_log_likelihood = tf.where(concept_samples == 1.,
                                        tf.math.log(concept_probs + FLAGS.epsilon),
                                        tf.math.log(1. - concept_probs + FLAGS.epsilon))
      latent_log_likelihood = tf.where(latent_samples == 1.,
                                       tf.math.log(latent_probs + FLAGS.epsilon),
                                       tf.math.log(1. - latent_probs + FLAGS.epsilon))
      return concept_samples, latent_samples, concept_log_likelihood, latent_log_likelihood
    else:
      if pretrain:
        intermediate = tf.random.normal([x.shape[0], FLAGS.intermediate_size])
      else:
        intermediate = self.predict_model(self.intermediate_predictor, self.get_normalized_inputs(x), self.x_shape,
                                          training=training)
      batch_shape = intermediate.shape[:-1]
      intermediate_broadcasted = tf.broadcast_to(intermediate, [n_samples] + intermediate.shape)
      concat_samples = tf.zeros(intermediate_broadcasted.shape[:-1] + [0], dtype=tf.float32)
      concept_log_likelihood = tf.zeros([n_samples] + batch_shape + [0], dtype=tf.float32)
      latent_log_likelihood = tf.zeros([n_samples] + batch_shape + [0], dtype=tf.float32)
      for i in range(self.n_concepts + self.n_latent_concepts):
        p_ci = self.predict_model(self.autoregressive_predictors[i],
                                  tf.concat((concat_samples, intermediate_broadcasted), axis=-1),
                                  input_shape=[i + FLAGS.intermediate_size],
                                  training=training)
        if i < self.n_concepts:
          i_samples = tf.where(c_mask[..., i:i + 1],
                               c[..., i:i + 1],
                               tf.where(tf.random.uniform(p_ci.shape) < p_ci,
                                        1.,
                                        0.))
          concept_log_likelihood = tf.concat((concept_log_likelihood,
                                              tf.where(i_samples == 1.,
                                                       tf.math.log(p_ci + FLAGS.epsilon),
                                                       tf.math.log(1. - p_ci + FLAGS.epsilon))), axis=-1)
        else:
          i_samples = tf.where(tf.random.uniform(p_ci.shape) < p_ci,
                               1.,
                               0.)
          latent_log_likelihood = tf.concat((latent_log_likelihood,
                                             tf.where(i_samples == 1.,
                                                      tf.math.log(p_ci + FLAGS.epsilon),
                                                      tf.math.log(1. - p_ci + FLAGS.epsilon))), axis=-1)

        concat_samples = tf.concat((concat_samples, i_samples), axis=-1)
      return concat_samples[..., :self.n_concepts], concat_samples[...,
                                                    self.n_concepts:], concept_log_likelihood, latent_log_likelihood

  def sample_p_c_z_given_x(self, x, n_samples, training=False):
    batch_shape = x.shape[:-len(list(self.x_shape))]
    return self.sample_p_c_z_given_x_masked(x,
                                            tf.zeros(batch_shape + [
                                              self.n_concepts],
                                                     dtype=tf.float32),
                                            tf.zeros(batch_shape + [
                                              self.n_concepts],
                                                     dtype=tf.bool),

                                            n_samples,
                                            training=training)

  def sample_p_z_given_x_c(self, x, c, n_samples, training=False):
    batch_shape = x.shape[:-len(list(self.x_shape))]
    _, z_samples, _, z_ll = self.sample_p_c_z_given_x_masked(x,
                                                             tf.cast(c, dtype=tf.float32),
                                                             tf.ones(batch_shape + [
                                                               self.n_concepts],
                                                                     dtype=tf.bool),
                                                             n_samples,
                                                             training=training)
    return z_samples, z_ll

  def p_c_z_given_x(self, x, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if not FLAGS.autoregressive:
      probs = self.predict_model(self.concept_predictor, self.get_normalized_inputs(x), self.x_shape, training=training)
      return probs[..., :self.n_concepts], probs[..., self.n_concepts:]
    else:
      c_samples, z_samples, c_ll, z_ll = self.sample_p_c_z_given_x(x,
                                                                   mc_samples,
                                                                   training=training)
      return tf.reduce_mean(c_samples, axis=0), tf.reduce_mean(z_samples, axis=0)

  def sample_p_z_given_c_masked(self, c, z, z_mask, n_samples, training=False):
    batch_shape = c.shape[:-1]
    if not FLAGS.autoregressive:
      latent_probs = self.predict_model(self.amortization_predictor, c, [self.n_concepts], training=training)
      z_samples = tf.where(z_mask,
                           z,
                           tf.where(tf.random.uniform([n_samples] + batch_shape + [self.n_latent_concepts],
                                                      dtype=tf.float32) < latent_probs,
                                    1., 0.))
      logprob = tf.where(z_samples == 1.,
                         tf.math.log(latent_probs + FLAGS.epsilon),
                         tf.math.log(1. - latent_probs + FLAGS.epsilon))
      return z_samples, logprob
    else:
      concat_samples = tf.zeros([n_samples] + batch_shape + [0], dtype=tf.float32)
      log_likelihood = tf.zeros([n_samples] + batch_shape + [0], dtype=tf.float32)
      c_broadcasted = tf.broadcast_to(tf.cast(c, dtype=tf.float32), [n_samples] + batch_shape + [self.n_concepts])
      for i in range(self.n_latent_concepts):
        p_zi = self.predict_model(self.autoregressive_amortizers[i],
                                  tf.concat((concat_samples, c_broadcasted), axis=-1),
                                  input_shape=[i + self.n_concepts],
                                  training=training)

        i_samples = tf.where(z_mask[..., i:i + 1],
                             z[..., i:i + 1],
                             tf.where(tf.random.uniform(p_zi.shape) < p_zi,
                                      1.,
                                      0.))
        log_likelihood = tf.concat((log_likelihood,
                                    tf.where(i_samples == 1.,
                                             tf.math.log(p_zi + FLAGS.epsilon),
                                             tf.math.log(1. - p_zi + FLAGS.epsilon))), axis=-1)

        concat_samples = tf.concat((concat_samples, i_samples), axis=-1)
      return concat_samples, log_likelihood

  def sample_p_z_given_c(self, c, n_samples, training=False):
    batch_shape = c.shape[:-1]
    return self.sample_p_z_given_c_masked(c,
                                          tf.zeros(batch_shape + [self.n_latent_concepts],
                                                   dtype=tf.float32),
                                          tf.zeros(batch_shape + [self.n_latent_concepts],
                                                   dtype=tf.bool), n_samples, training=training)

  def p_z_given_c(self, c, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if not FLAGS.autoregressive:
      return self.predict_model(self.amortization_predictor, c, [self.n_concepts], training=training)
    else:
      z_samples, z_ll = self.sample_p_z_given_c(c,
                                                mc_samples,
                                                training=training)
      return tf.reduce_mean(z_samples, axis=0)

  def p_y_given_c_z(self, c, z, training=False):
    if self.type == 'joint' or self.type == 'sequential':
      return self.predict_model(self.label_predictor,
                                logit(tf.concat((tf.cast(c, dtype=tf.float32),
                                           tf.cast(z, dtype=tf.float32)),
                                          axis=-1)),
                                [self.n_concepts + self.n_latent_concepts],
                                training=training)
    else:
      return self.predict_model(self.label_predictor,
                                tf.concat((tf.cast(c, dtype=tf.float32),
                                           tf.cast(z, dtype=tf.float32)),
                                          axis=-1) - 0.5,
                                [self.n_concepts + self.n_latent_concepts],
                                training=training)

  def p_y_given_x(self, x, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if self.soft:
      assert not FLAGS.autoregressive
      c_probs, z_probs = self.p_c_z_given_x(x, training=training)
      return self.p_y_given_c_z(c_probs, z_probs, training=training)
    else:
      c_samples, z_samples, c_ll, z_ll = self.sample_p_c_z_given_x(x,
                                                                   mc_samples,
                                                                   training=training)
      return reinforce_mean(self.p_y_given_c_z(c_samples, z_samples),
                            tf.reduce_sum(tf.concat((c_ll, z_ll), axis=-1), axis=-1))

  def p_y_given_c(self, c, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if self.soft:
      assert not FLAGS.autoregressive
      z_probs = self.p_z_given_c(c, training=training)
      return self.p_y_given_c_z(c, z_probs, training=training)
    else:
      z_samples, z_ll = self.sample_p_z_given_c(c,
                                                mc_samples,
                                                training=training)
      c_broadcasted = tf.broadcast_to(tf.cast(c, tf.float32), [mc_samples] + c.shape)
      return reinforce_mean(self.p_y_given_c_z(c_broadcasted, z_samples),
                            tf.reduce_sum(z_ll, axis=-1))

  def p_y_given_x_markovian(self, x, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if self.soft:
      assert not FLAGS.autoregressive
      c_probs, _ = self.p_c_z_given_x(x, training=training)
      z_probs = self.p_z_given_c(c_probs, training=training)
      return self.p_y_given_c_z(c_probs, z_probs, training=training)
    else:
      c_samples, _, c_ll, _ = self.sample_p_c_z_given_x(x,
                                                        mc_samples,
                                                        training=training)
      z_samples, z_ll = self.sample_p_z_given_c(c_samples, 1, training=training)
      z_samples = z_samples[0, ...]
      z_ll = z_ll[0, ...]
      return reinforce_mean(self.p_y_given_c_z(c_samples, z_samples),
                            tf.reduce_sum(tf.concat((c_ll, z_ll), axis=-1), axis=-1))

  def p_y_given_x_c(self, x, c, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if self.soft:
      assert not FLAGS.autoregressive
      c_probs, z_probs = self.p_c_z_given_x(x, training=training)
      return self.p_y_given_c_z(c, z_probs, training=training)
    else:
      z_samples, z_ll = self.sample_p_z_given_x_c(x, c,
                                                  mc_samples,
                                                  training=training)
      c_broadcasted = tf.broadcast_to(tf.cast(c, dtype=tf.float32), [mc_samples] + c.shape)
      return reinforce_mean(self.p_y_given_c_z(c_broadcasted, z_samples), tf.reduce_sum(z_ll, axis=-1))

  def p_y_given_x_intervention(self, x, c, intervention_mask, training=False):
    mc_samples = FLAGS.mc_samples_for_training if training else FLAGS.mc_samples_for_prediction
    if self.soft:
      assert not FLAGS.autoregressive
      c_probs, z_probs = self.p_c_z_given_x(x, training=training)
      soft_c = tf.where(intervention_mask == 1.,
                        tf.cast(c, dtype=tf.float32),
                        c_probs)
      return self.p_y_given_c_z(soft_c, z_probs, training=training)
    else:
      c_samples, z_samples, c_ll, z_ll = self.sample_p_c_z_given_x_masked(x,
                                                                          tf.cast(c, dtype=tf.float32),
                                                                          tf.broadcast_to(intervention_mask == 1.,
                                                                                          c.shape),
                                                                          mc_samples, training=training)
      log_weights = tf.reduce_sum(tf.where(intervention_mask == 1., c_ll, 0.), axis=-1)
      weights = tf.nn.softmax(log_weights, axis=0)
      return tf.reduce_sum(self.p_y_given_c_z(c_samples, z_samples) * weights[:, :, None], axis=0)

  # def p_z_given_c(self, c):
  #   input_shape = c.shape
  #   batch_shape = input_shape[:-1]
  #   inputs = tf.reshape(c, [np.prod(batch_shape, dtype=np.int32)] + input_shape[-1:])
  #   result = self.amortization_predictor(tf.cast(inputs, dtype=tf.float32))
  #   return tf.reshape(result, batch_shape + result.shape[-1])
  #
  # def p_y_given_c_z(self, c, z):
  #   inputs = tf.concat((c, z), axis=-1)
  #   input_shape = inputs.shape
  #   batch_shape = input_shape[:-1]
  #   inputs = tf.reshape(inputs, [np.prod(batch_shape, dtype=np.int32)] + input_shape[-1:])
  #   result = self.label_predictor(tf.cast(inputs, dtype=tf.float32))
  #   return tf.reshape(result, batch_shape + result.shape[-1])
  #
  # def p_y_given_c_x(self, c, x):
  #   z_dist = tfd.Bernoulli(probs=self.p_c_z_given_x(x)[1], validate_args=True)
  #   z_samples = z_dist.sample(FLAGS.mc_samples_for_prediction)
  #   z_samples = tf.cast(z_samples, tf.float32)
  #   c_broadcasted = tf.broadcast_to(tf.cast(c, dtype=tf.float32), [FLAGS.mc_samples_for_prediction] + c.shape)
  #   z_logprob = tf.reduce_sum(z_dist.log_prob(z_samples), axis=-1)
  #   return reinforce_mean(self.p_y_given_c_z(c_broadcasted, z_samples), z_logprob)
  #
  # def p_y_given_c(self, c):
  #   z_samples = tfd.Bernoulli(probs=self.p_z_given_c(c), validate_args=True).sample(FLAGS.mc_samples_for_prediction)
  #   z_samples = tf.cast(z_samples, tf.float32)
  #   c_broadcasted = tf.broadcast_to(tf.cast(c, dtype=tf.float32), [FLAGS.mc_samples_for_prediction] + c.shape)
  #   return tf.reduce_mean(self.p_y_given_c_z(c_broadcasted, z_samples), axis=0)
  #
  # def p_y_given_x(self, x, training=False):
  #   input_shape = x.shape
  #   batch_shape = input_shape[:-len(self.input_shape_list)]
  #   inputs = tf.reshape(x, [np.prod(batch_shape, dtype=np.int32)] + self.input_shape_list)
  #
  #   c_probs, z_probs = self.p_c_z_given_x(inputs)
  #   c_dist = tfd.Bernoulli(probs=c_probs, validate_args=True)
  #   c_samples = c_dist.sample([FLAGS.mc_samples_for_prediction])
  #   c_logprob = tf.reduce_sum(c_dist.log_prob(c_samples), axis=-1)
  #   z_dist = tfd.Bernoulli(probs=z_probs, validate_args=True)
  #   z_samples = z_dist.sample([FLAGS.mc_samples_for_prediction])
  #   z_logprob = tf.reduce_sum(z_dist.log_prob(z_samples), axis=-1)
  #
  #   y_probs = self.p_y_given_c_z(c_samples, z_samples)
  #
  #   y_probs = reinforce_mean(y_probs, c_logprob + z_logprob)
  #   return tf.reshape(y_probs, batch_shape + y_probs.shape[-1])
  #
  # def p_y_given_x_markovian(self, x, training=False):
  #   input_shape = x.shape
  #   batch_shape = input_shape[:-len(self.input_shape_list)]
  #   inputs = tf.reshape(x, [np.prod(batch_shape, dtype=np.int32)] + self.input_shape_list)
  #
  #   c_dist = tfd.Bernoulli(probs=self.p_c_z_given_x(inputs)[0], validate_args=True)
  #   c_samples = c_dist.sample([FLAGS.mc_samples_for_prediction])
  #   c_logprob = tf.reduce_sum(c_dist.log_prob(c_samples), axis=-1)
  #   y_probs = self.p_y_given_c(c_samples)
  #   y_probs = reinforce_mean(y_probs, c_logprob)
  #   return tf.reshape(y_probs, batch_shape + y_probs.shape[-1])

  def get_normalized_inputs(self, inputs):
    return (inputs - self.input_mean) / self.input_std

  #
  # def get_unnormalized_inputs(self, inputs):
  #   return inputs * self.input_std + self.input_mean

  def train(self, optimizer_name, dataset, epochs, batch_size, learning_rate, amortization_epochs, burn_in=False,
            soft_gradient='independent', label_lr_multiplier=1., pretrain_autoregressive=0):
    dataset_size = tf.data.experimental.cardinality(dataset)
    batched_dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if optimizer_name == 'sghmc':
      if self.optimizer is None or burn_in:
        self.optimizer = sghmc.AdaptiveSGHMC(learning_rate=learning_rate,
                                             burnin=tf.data.experimental.cardinality(batched_dataset) * epochs,
                                             data_size=dataset_size,
                                             overestimation_rate=1,
                                             initialization_rounds=10,
                                             friction=0.05)
    elif optimizer_name == 'adam':
      self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        name='adam_optimizer'
      )
    elif optimizer_name == 'sgd':
      total_steps = epochs * dataset_size // batch_size
      lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=total_steps // 3,
        decay_rate=0.1,
        staircase=True)
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, name='sgd_optimizer', nesterov=True)

    self.label_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        name='label_optimizer'
      )
    self.pretrain_autoregressive_optimizer = tf.keras.optimizers.Adam(
      learning_rate=learning_rate,
      name='pretrain_adam_optimizer'
    )
    def loss_fn(batch, pretrain=False):
      x = batch['data']
      y = batch['label']
      c = tf.cast(batch['concepts'], dtype=tf.float32)
      if self.soft:
        concept_probs, latent_probs = self.p_c_z_given_x(x, training=True)
        concept_loss = -tf.reduce_mean(tf.reduce_sum(tf.where(c == 1,
                                                              tf.math.log(concept_probs + FLAGS.epsilon),
                                                              tf.math.log(1. - concept_probs + FLAGS.epsilon)),
                                                     axis=-1))
        if soft_gradient == 'independent':
          y_probs = self.p_y_given_c_z(tf.cast(c, dtype=tf.float32), latent_probs)
        elif soft_gradient == 'sequential':
          y_probs = self.p_y_given_c_z(tf.stop_gradient(concept_probs), latent_probs)
        elif soft_gradient == 'joint':
          y_probs = self.p_y_given_c_z(concept_probs, latent_probs)
      else:
        batch_shape = c.shape[:-1]
        mc_samples = 1 if self.n_latent_concepts == 0 else FLAGS.mc_samples_for_training
        c_samples, z_samples, c_ll, z_ll = self.sample_p_c_z_given_x_masked(x, c,
                                                                            tf.ones(batch_shape + [self.n_concepts],
                                                                                    dtype=tf.bool),
                                                                            mc_samples,
                                                                            training=True,
                                                                            pretrain=pretrain)
        concept_log_likelihoods = tf.reduce_mean(tf.reduce_sum(c_ll, axis=-1), axis=0)
        concept_loss = -tf.reduce_mean(concept_log_likelihoods)

        y_probs = reinforce_mean(self.p_y_given_c_z(c_samples, z_samples), tf.reduce_sum(z_ll, axis=-1))
      label_log_likelihood = tf.keras.losses.categorical_crossentropy(
        tf.one_hot(y, depth=self.n_label_classes),
        y_probs)
      label_loss = tf.reduce_mean(label_log_likelihood)
      prior_loss = self.get_prior_log_likelihood()
      if not self.use_inceptionv3:
        prior_loss = -prior_loss / tf.cast(dataset_size, tf.float32)
      return concept_loss, label_loss, prior_loss

    @tf.function
    def amortization_loss(batch):
      x = batch['data']
      c = tf.cast(batch['concepts'], dtype=tf.float32)

      batch_shape = c.shape[:-1]
      c_samples, z_samples, c_ll, z_ll = self.sample_p_c_z_given_x_masked(x,
                                                                          c,
                                                                          tf.ones(batch_shape + [self.n_concepts],
                                                                                  dtype=tf.bool),
                                                                          FLAGS.mc_samples_for_training,
                                                                          training=False)
      z_samples, z_ll = self.sample_p_z_given_c_masked(tf.broadcast_to(c, [FLAGS.mc_samples_for_training] + c.shape),
                                                       z_samples,
                                                       tf.ones(
                                                         [FLAGS.mc_samples_for_training] + batch_shape + [
                                                           self.n_latent_concepts], dtype=tf.bool), 1, training=True)
      loss = -tf.reduce_mean(tf.reduce_sum(z_ll[0, ...], axis=-1), axis=[0, 1])
      return loss

    if FLAGS.autoregressive:
      concept_parameters = self.intermediate_predictor.trainable_weights
      autoregressive_parameters = []
      for a in self.autoregressive_predictors:
        concept_parameters = concept_parameters + a.trainable_weights
        autoregressive_parameters = autoregressive_parameters + a.trainable_weights
    else:
      concept_parameters = self.concept_predictor.trainable_weights
    for epoch in range(epochs + pretrain_autoregressive):
      epoch_start = time.time()
      with tqdm(enumerate(batched_dataset), total=int(tf.data.experimental.cardinality(batched_dataset)),
                desc='Epoch {}'.format(epoch)) as pbar:
        moving_loss = tf.keras.metrics.Mean()
        moving_concept_ll = tf.keras.metrics.Mean()
        moving_label_ll = tf.keras.metrics.Mean()
        moving_prior_ll = tf.keras.metrics.Mean()
        for step, batch in pbar:
          with tf.GradientTape() as tape:
            concept_ll, label_ll, prior_ll = loss_fn(batch, pretrain=False)
            loss = FLAGS.lamb * concept_ll + label_lr_multiplier * label_ll + prior_ll
          if epoch < pretrain_autoregressive:
            grads = tape.gradient(loss, autoregressive_parameters + self.label_predictor.trainable_weights)
            self.pretrain_autoregressive_optimizer.apply_gradients(zip(grads, autoregressive_parameters + self.label_predictor.trainable_weights))
          else:
            grads = tape.gradient(loss, concept_parameters + self.label_predictor.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, concept_parameters + self.label_predictor.trainable_weights))


          moving_loss.update_state(loss)
          moving_concept_ll.update_state(concept_ll)
          moving_label_ll.update_state(label_ll)
          moving_prior_ll.update_state(prior_ll)
          pbar.set_postfix({'L': "%0.4f" % moving_loss.result().numpy(),
                            'Concept LL': "%0.4f" % moving_concept_ll.result().numpy(),
                            'Label LL': "%0.4f" % moving_label_ll.result().numpy(),
                            'Prior LL': "%0.4f" % moving_prior_ll.result().numpy()})
      print(f'Epoch time {time.time() - epoch_start}')

    if not burn_in and self.n_latent_concepts > 0:
      self.amortization_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        name='amortization_optimizer'
      )

      if FLAGS.autoregressive:
        trainable_parameters = []
        for a in self.autoregressive_amortizers:
          trainable_parameters = trainable_parameters + a.trainable_weights
      else:
        trainable_parameters = self.amortization_predictor.trainable_weights
      for epoch in range(amortization_epochs if amortization_epochs is not None else epochs):
        with tqdm(enumerate(batched_dataset), total=int(tf.data.experimental.cardinality(batched_dataset)),
                  desc='Epoch {} (Amortization network)'.format(epoch)) as pbar:
          moving_amort_loss = tf.keras.metrics.Mean()
          for step, batch in pbar:
            with tf.GradientTape() as tape:
              amort_loss = amortization_loss(batch)
            grads = tape.gradient(amort_loss, trainable_parameters)
            self.amortization_optimizer.apply_gradients(zip(grads, trainable_parameters))

            moving_amort_loss.update_state(amort_loss)
            pbar.set_postfix({'Amort L': "%0.4f" % moving_amort_loss.result().numpy()})

  def latent_concept_importance(self, dataset, n_mc_samples, batch_size=512):
    dataset_size = int(tf.data.experimental.cardinality(dataset))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def accumulator_fn(batch):
      x = batch['data']
      c = batch['concepts']
      y = batch['label']

      z_given_x = self.p_z_given_x(x)
      z_given_c = self.p_z_given_c(c)
      sample_z = tf.cast(tfd.Bernoulli(probs=z_given_c, validate_args=True).sample(n_mc_samples),
                         tf.float32)  # mc x N x latent
      y_pred = self.p_y_given_c_z(tf.broadcast_to(c, [n_mc_samples] + c.shape), sample_z)  # mc x N x 1
      weights = tf.where(sample_z == 1.,
                         z_given_x / z_given_c,
                         (1. - z_given_x) / (1. - z_given_c))  # mc x N x latent
      weighted_predictions = tf.reduce_sum(y_pred[:, :, :, None] * weights[:, :, None, :], axis=0) / \
                             tf.reduce_sum(weights, axis=0)[:, None, :]  # N x 1 x latent
      log_likelihoods = tf.where(y[:, None, None] == 1.,
                                 tf.math.log(weighted_predictions),
                                 tf.math.log(1. - weighted_predictions))  # N x 1 x latent
      return tf.reduce_sum(tf.reduce_sum(log_likelihoods, axis=1), axis=0)

    latent_log_likelihoods = tf.zeros((self.n_latent_concepts,), tf.float32)
    for batch in tqdm(dataset,
                      total=int(tf.data.experimental.cardinality(dataset)),
                      desc='Calculating latent feature importance'):
      latent_log_likelihoods = latent_log_likelihoods + accumulator_fn(batch)
    latent_log_likelihoods = latent_log_likelihoods / dataset_size
    return tf.argsort(latent_log_likelihoods, direction='DESCENDING')

  def top_disagreeing_pairs(self, key_latent, k, dataset, batch_size=512, batched_dataset=None):
    index_array = np.array(list(itertools.product([0, 1], repeat=self.n_concepts)), np.float32)
    if batched_dataset is None:
      dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
      dataset = batched_dataset
    top_positive_prob = tf.zeros((index_array.shape[0], k), tf.float32)
    top_positive_input = tf.zeros((index_array.shape[0], k, self.n_features), tf.float32)

    top_negative_prob = tf.zeros((index_array.shape[0], k), tf.float32)
    top_negative_input = tf.zeros((index_array.shape[0], k, self.n_features), tf.float32)

    @tf.function
    def accumulator_fn(batch, top_positive_prob, top_positive_input, top_negative_prob, top_negative_input):
      x = batch['data']
      c = batch['concepts']

      z_given_x = self.p_z_given_x(x)
      positive_prob = tf.where(tf.reduce_all(index_array[:, None, :] == c[None, :, :], axis=-1),
                               z_given_x[None, :, key_latent],
                               0.)  # 2^n_concept x N
      positive_prob_concat = tf.concat((top_positive_prob, positive_prob), axis=-1)
      top_positive_indices = tf.math.top_k(positive_prob_concat, k, sorted=True)
      top_positive_indices = top_positive_indices.indices
      top_positive_prob = tf.gather(positive_prob_concat, top_positive_indices, axis=-1, batch_dims=1)
      positive_input_concat = tf.concat((top_positive_input, tf.broadcast_to(x, [index_array.shape[0]] + x.shape)),
                                        axis=1)
      top_positive_input = tf.gather(positive_input_concat, top_positive_indices, axis=1, batch_dims=1)

      negative_prob = tf.where(tf.reduce_all(index_array[:, None, :] == c[None, :, :], axis=-1),
                               1. - z_given_x[None, :, key_latent],
                               0.)  # 2^n_concept x N
      negative_prob_concat = tf.concat((top_negative_prob, negative_prob), axis=-1)
      top_negative_indices = tf.math.top_k(negative_prob_concat, k, sorted=True)
      top_negative_indices = top_negative_indices.indices
      top_negative_prob = tf.gather(negative_prob_concat, top_negative_indices, axis=-1, batch_dims=1)
      negative_input_concat = tf.concat((top_negative_input, tf.broadcast_to(x, [index_array.shape[0]] + x.shape)),
                                        axis=1)
      top_negative_input = tf.gather(negative_input_concat, top_negative_indices, axis=1, batch_dims=1)

      return top_positive_prob, top_positive_input, top_negative_prob, top_negative_input

    for batch in tqdm(dataset,
                      total=int(tf.data.experimental.cardinality(dataset)),
                      desc='Finding disagreeing pairs.'):
      top_positive_prob, top_positive_input, top_negative_prob, top_negative_input = accumulator_fn(batch,
                                                                                                    top_positive_prob,
                                                                                                    top_positive_input,
                                                                                                    top_negative_prob,
                                                                                                    top_negative_input)

    # Tested until this point.
    top_positive_prob = tf.reshape(top_positive_prob, [-1])
    sorted_positive_indices = tf.argsort(top_positive_prob, direction='DESCENDING')
    top_positive_prob = tf.gather(top_positive_prob, sorted_positive_indices)
    top_positive_input = tf.reshape(top_positive_input, [-1, self.n_features])
    top_positive_input = tf.gather(top_positive_input, sorted_positive_indices, axis=0)
    top_negative_prob = tf.reshape(top_negative_prob, [-1])
    sorted_negative_indices = tf.argsort(top_negative_prob, direction='DESCENDING')
    top_negative_prob = tf.gather(top_negative_prob, sorted_positive_indices)
    top_negative_input = tf.reshape(top_negative_input, [-1, self.n_features])
    top_negative_input = tf.gather(top_negative_input, sorted_negative_indices, axis=0)
    best_pair_probs = []
    best_pair_inputs = []
    positive_index = 0
    negative_index = 0
    for _ in range(k):
      best_pair_probs.append((top_positive_prob[positive_index], top_negative_prob[negative_index]))
      best_pair_inputs.append((top_positive_input[positive_index], top_negative_input[negative_index]))
      if (top_positive_prob[positive_index + 1] * top_negative_prob[negative_index] >
          top_positive_prob[positive_index] * top_negative_prob[negative_index + 1]):
        positive_index += 1
      else:
        negative_index += 1

    return best_pair_inputs
