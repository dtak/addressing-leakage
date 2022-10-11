"""
Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to make a validation set
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
"""

import os
import numpy as np
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict


def extract_data(data_dir):
  data_path = data_dir + '/images'

  path_to_id_map = dict()  # map from full image path to image id
  with open(data_path.replace('images', 'images.txt'), 'r') as f:
    for line in f:
      items = line.strip().split()
      path_to_id_map[join(data_path, items[1])] = int(items[0])

  attribute_labels_all = ddict(list)  # map from image id to a list of attribute labels
  attribute_certainties_all = ddict(list)  # map from image id to a list of attribute certainties
  attribute_uncertain_labels_all = ddict(
    list)  # map from image id to a list of attribute labels calibrated for uncertainty
  # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
  uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4: 1},  # calibrate main label based on uncertainty label
                     0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
  with open(data_dir + '/attributes/image_attribute_labels.txt', 'r') as f:
    for line in f:
      file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
      attribute_label = int(attribute_label)
      attribute_certainty = int(attribute_certainty)
      uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
      attribute_labels_all[int(file_idx)].append(attribute_label)
      attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
      attribute_certainties_all[int(file_idx)].append(attribute_certainty)

  is_train_test = dict()  # map from image id to 0 / 1 (1 = train)
  with open(data_dir + '/train_test_split.txt', 'r') as f:
    for line in f:
      idx, is_train = line.strip().split()
      is_train_test[int(idx)] = int(is_train)
  print("Number of train images from official train test split:", sum(list(is_train_test.values())))

  train_data = []
  folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
  folder_list.sort()  # sort by class index
  for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')]
    # classfile_list.sort()
    for cf in classfile_list:
      img_id = path_to_id_map[join(folder_path, cf)]
      img_path = join(folder_path, cf)
      metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                  'attribute_label': attribute_labels_all[img_id],
                  'attribute_certainty': attribute_certainties_all[img_id],
                  'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
      if is_train_test[img_id]:
        train_data.append(metadata)

  return train_data

def get_class_attributes(data):
  class_attr_count = np.zeros((200, 312, 2))
  for d in data:
    class_label = d['class_label']
    certainties = d['attribute_certainty']
    for attr_idx, a in enumerate(d['attribute_label']):
      if a == 0 and certainties[attr_idx] == 1:  # not visible
        continue
      class_attr_count[class_label][attr_idx][a] += 1

  class_attr_min_label = np.argmin(class_attr_count, axis=2)
  class_attr_max_label = np.argmax(class_attr_count, axis=2)
  equal_count = np.where(
    class_attr_min_label == class_attr_max_label)  # check where 0 count = 1 count, set the corresponding class attribute label to be 1
  class_attr_max_label[equal_count] = 1

  attr_class_count = np.sum(class_attr_max_label, axis=0)
  mask = np.where(attr_class_count >= 10)[0]  # select attributes that are present (on a class level) in at least [min_class_count] classes
  print(mask)
  consensus_threshold = 0.9
  consensus_mask = np.where(np.sum(
    np.where(class_attr_max_label == 1, class_attr_count[:, :, 1], class_attr_count[:, :, 0]), axis=0)
                            / np.sum(class_attr_count[:, :, 1] + class_attr_count[:, :, 0], axis=0) >= consensus_threshold
  )[0]
  print(consensus_mask)
  mask = [ind for ind in mask if ind in consensus_mask]
  # Use mask provided by https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/generate_new_data.py
  mask = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
  print(mask) 
  class_attr_label_masked = class_attr_max_label[:, mask]
  return class_attr_label_masked, mask

if __name__ == "__main__":
  dir = 'CUB_200_2011'
  train_data = extract_data(dir)

  class_concepts, mask = get_class_attributes(train_data)
  np.save('attributes_and_labels/class_concepts.npy',
          class_concepts)

  concept_names = []
  with open('attributes_and_labels/attributes.txt', 'r') as f:
    for line in f:
      concept_names.append(line.replace('\n', '').split(' ')[1])
  f = open("attributes_and_labels/concept_names.txt", "w")
  f.writelines([concept_names[i] + '\n' for i in mask])
  f.close()
