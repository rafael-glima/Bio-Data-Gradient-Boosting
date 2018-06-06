import numpy as np
#import tensorflow as tf
#import keras
#from keras.layers import *
np.random.seed(2018)
import random
random.seed(2018)

from data.preprocess_data import preprocess_data
#from model.network import my_model
import config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import pydotplus
from PIL import Image
#import graphviz

# CONSTANT
LEARNING_RATE = config.learning_rate
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs

# Load data
file_path = 'data.npy'
y_train, x_train_feature, x_train_ref, x_train_tumor,\
y_valid, x_valid_feature, x_valid_ref, x_valid_tumor,\
y_test, x_test_feature, x_test_ref, x_test_tumor = preprocess_data(file_path)

def one_hot_to_indices(data):
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices

y_train_labels = one_hot_to_indices(y_train)
y_test_labels = one_hot_to_indices(y_test)

#print(y_train_labels.shape)
#print(y_test_labels.shape)

train_features = np.reshape(x_train_feature,(x_train_feature.shape[0],x_train_feature.shape[1]))
test_features = np.reshape(x_test_feature,(x_test_feature.shape[0],x_test_feature.shape[1]))

print(train_features.shape)
print(test_features.shape)

gb = GradientBoostingClassifier(n_estimators=10,max_depth=1,verbose=1)
gb.fit(train_features,y_train_labels)

print(gb.predict(test_features)[:100])
print(y_test_labels[:100])
gb.score(test_features,y_test_labels)

print(gb.score(test_features,y_test_labels))

# # Get the tree number 2
# sub_tree_2 = gb.estimators_[2, 0]

# dot_data = tree.export_graphviz(
#     sub_tree_2,
#     out_file=None, filled=True,
#     rounded=True,  
#     special_characters=True,
#     proportion=True,
# )
# graph = pydotplus.graph_from_dot_data(dot_data)  
# Image(graph.create_png()) 