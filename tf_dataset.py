import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pathlib
import os
from absl import app
from absl import flags
import csv

FLAGS = flags.FLAGS

flags.DEFINE_string('class_concepts_path', '/n/home10/mhavasi/concept_prediction/data/cub/class_concepts.npy', 'Npy file containing the concepts for each label.')

image_size = 299
num_classes = 200

def get_concept_groups():
  concept_names = []
  with open('/n/home10/mhavasi/concept_prediction/data/cub/concept_names.txt', 'r') as f:
    for line in f:
      concept_names.append(line.replace('\n', '').split('::'))

  group_names = []
  for c in concept_names:
    if c[0] not in group_names:
      group_names.append(c[0])
  groups = np.zeros((len(group_names), len(concept_names)), dtype=np.float32)
  for i, gn in enumerate(group_names):
    for j, cn in enumerate(concept_names):
      if cn[0] == gn:
        groups[i, j] = 1.
  return groups

def get_dataset(train, augmented):
  class_concepts = tf.constant(np.load(FLAGS.class_concepts_path))
  def image_preproces_train(input):
    image = input['image']
    image = tf.cast(image, tf.float32)
    seed = tf.random.uniform((2,), 0, 1000000, dtype=tf.int32)
    scale = tf.random.uniform((), 0.08, 1.0, dtype=tf.float32)
    image_shape = tf.shape(image)
    area = tf.cast(image_shape[0] * image_shape[1], dtype=tf.float32)
    rescaled_area = area * scale
    aspect_ratio = tf.random.uniform((), 0.75, 1.333, dtype=tf.float32)
    new_aspect_ratio = aspect_ratio * tf.cast(image_shape[0], dtype=tf.float32) / tf.cast(image_shape[1], dtype=tf.float32)
    desired_size = (tf.math.sqrt(rescaled_area * new_aspect_ratio), tf.math.sqrt(rescaled_area / new_aspect_ratio))
    new_size = [tf.minimum(tf.cast(desired_size[0], dtype=tf.int32), image_shape[0]),
                tf.minimum(tf.cast(desired_size[1], dtype=tf.int32), image_shape[1]),
                3]
    image = tf.image.stateless_random_crop(image, new_size, seed=seed)
    image = tf.image.resize_with_pad(image, tf.cast(desired_size[0], tf.int32), tf.cast(desired_size[1], tf.int32))
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_brightness(
      image, max_delta=32. / 255., seed=seed)
    image = tf.image.stateless_random_saturation(
      image, lower=0.5, upper=1.5, seed=seed)
    image = (image / 127.5) - 1.
    image = tf.clip_by_value(image, -1, 1)
    output = {}
    output['data'] = image
    output['label'] = input['label']
    output['concepts'] = class_concepts[input['label'], :]
    return output


  def image_preproces_test(input):
    image = input['image']
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = (image / 127.5) - 1.
    image = tf.clip_by_value(image, -1, 1)
    output = {}
    output['data'] = image
    output['label'] = input['label']
    output['concepts'] = class_concepts[input['label'], :]
    return output


  if train:
    dataset = tfds.load('caltech_birds2011', split='train', shuffle_files=True).cache()
  else:
    dataset = tfds.load('caltech_birds2011', split='test').cache()

  if augmented:
    return dataset.map(image_preproces_train)
  else:
    return dataset.map(image_preproces_test)
