## Utils.py -- Some utility functions 
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from tensorflow.contrib.keras.api.keras.models import Model, model_from_json, Sequential
from PIL import Image

import tensorflow as tf
import os
import numpy as np

def generate_attack_data_set(data, num_sample, img_offset, model, attack_type="targeted", random_target_class=None, shift_index=False):
    """
    Generate the data for conducting attack. Only select the data being classified correctly.
    """
    orig_img = []
    orig_labels = []
    target_labels = []
    orig_img_id = []

    pred_labels = np.argmax(model.model.predict(data.test_data), axis=1)
    true_labels = np.argmax(data.test_labels, axis=1)
    correct_data_indices = np.where([1 if x==y else 0 for (x,y) in zip(pred_labels, true_labels)])

    print("Total testing data:{}, correct classified data:{}".format(len(data.test_labels), len(correct_data_indices[0])))

    data.test_data = data.test_data[correct_data_indices]
    data.test_labels = data.test_labels[correct_data_indices]
    true_labels = true_labels[correct_data_indices]


    np.random.seed(img_offset) # for parallel running
    class_num = data.test_labels.shape[1]
    for sample_index in range(num_sample):

        if attack_type == "targeted":
            if random_target_class is not None:
                np.random.seed(0)  # for parallel running
                # randomly select one class to attack, except the true labels
                # print(random_target_class)
                seq_imagenet = np.random.choice(random_target_class, 100)
                seq = [seq_imagenet[img_offset + sample_index]]
                # seq = np.random.choice(random_target_class, 1)
                while seq == true_labels[img_offset+sample_index]:
                    seq = np.random.choice(random_target_class, 1)
                
            else:
                seq = list(range(class_num))
                seq.remove(true_labels[img_offset+sample_index])

            for s in seq:
                if shift_index and s == 0:
                    s += 1
                orig_img.append(data.test_data[img_offset+sample_index])
                target_labels.append(np.eye(class_num)[s])
                orig_labels.append(data.test_labels[img_offset+sample_index])
                orig_img_id.append(img_offset+sample_index)

        elif attack_type == "untargeted":
            orig_img.append(data.test_data[img_offset+sample_index])
            target_labels.append(data.test_labels[img_offset+sample_index])
            orig_labels.append(data.test_labels[img_offset+sample_index])
            orig_img_id.append(img_offset+sample_index)

    orig_img = np.array(orig_img)
    target_labels = np.array(target_labels)
    orig_labels = np.array(orig_labels)
    orig_img_id = np.array(orig_img_id)

    return orig_img, target_labels, orig_labels, orig_img_id

def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n','')
    return prob, predicted_class, prob_str
