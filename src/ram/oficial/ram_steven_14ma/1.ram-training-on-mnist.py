#!/usr/bin/env python
# coding: utf-8

# # Training RAM on MNIST
# 
# ## Requirements
# 
# ### Imports

# In[8]:


from tqdm import tqdm

# 2019041500 - use this tf nightly version
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

#from model.ram_tf_1_13 import RecurrentAttentionModel
from model.ram import RecurrentAttentionModel

from data.augmented_mnist import minibatcher
from data.augmented_mnist import get_mnist

from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs


# ### Data

# In[3]:


batch_size = 100

(X_train, y_train),(X_test, y_test) = get_mnist(True, True, False)

print(X_train.shape, y_train.shape, np.max(X_train), np.min(X_train))
print(X_test.shape, y_test.shape, np.max(X_test), np.min(X_test))


# ## Training

# ### Hyperparameter

# In[ ]:


learning_rate = 0.001
std = 0.25

ram = RecurrentAttentionModel(time_steps=7,
                              n_glimpses=1, 
                              glimpse_size=8,
                              num_classes=10,
                              max_gradient_norm=1.0,
                              std=std)
adam_opt = tf.keras.optimizers.Adam(learning_rate)


# ### Trainingsloop

# In[ ]:


for timestep in tqdm(range(200)): # after 50 steps a accuracy of 95% is reached 
    losses = []
    rewards = []
    classification_losses = []
    
    # training steps
    batcher = minibatcher(X_train, y_train, batch_size, True)

    for X, y in batcher:
        with tf.GradientTape() as tape:
            # calculate lossest
            logits = ram(X)
            
            loss, classification_loss, reward, _ = ram.hybrid_loss(logits, y)
            
            # append to list for output
            losses.append(loss.numpy())
            classification_losses.append(classification_loss.numpy())
            rewards.append(reward.numpy())
            
            # calculate gradient and do gradient descent
            gradients = tape.gradient(loss, ram.trainable_variables)
            adam_opt.apply_gradients(zip(gradients, ram.trainable_variables))
        
    # testing steps
    batcher = minibatcher(X_test, y_test, batch_size, True)
    accuracys = []
    for X, y in batcher:
        logits = ram(X)
        accuracy, prediction, location = ram.predict(logits, y)
        accuracys.append(accuracy.numpy())

    print("step", timestep, "accuracy:", np.mean(accuracys))


# ## Testing

# In[ ]:


batcher = minibatcher(X_test, y_test, batch_size, True)
accuracys = []
for X, y in batcher:
    logits = ram(X)
    accuracy, prediction, location = ram.predict(logits, y)
    accuracys.append(accuracy.numpy())

print("accuracy:", np.mean(accuracys))


# In[ ]:


for i in range(10):
    index = np.where(np.argmax(y_test, 1) == i)[0]
    batcher = minibatcher(X_test[index], y_test[index], batch_size, True)
    accuracys = []
    for X, y in batcher:
        logits = ram(X)
        accuracy, prediction, location = ram.predict(logits, y)
        accuracys.append(accuracy.numpy())
    print("number", i, "accuracy:", np.mean(accuracys))


# ## Visualization

# In[ ]:


def plot_path_of(number, batch):
    from visualization.model import plot_prediction_path
    imgs = X_test[batch*batch_size:batch*batch_size + batch_size]
    labels = y_test[batch*batch_size:batch*batch_size + batch_size]
    logits = ram(imgs)
    _, prediction, location = ram.predict(logits, labels)
    labels = np.argmax(labels, 1)
    for i, (y, y_hat) in enumerate(zip(list(prediction.numpy()), list(labels))):
        if y == y_hat & y == number:
            loc = location[i].numpy()
            img = imgs[i]
            plot_prediction_path(img, loc, 1, 8)
        if y != y_hat & y == number:
            loc = location[i].numpy()
            img = imgs[i]
            plot_prediction_path(img, loc, 1, 8)


# In[ ]:


plot_path_of(0, 1)
plot_path_of(0, 13)
plot_path_of(0, 7)
plot_path_of(0, 5)
plot_path_of(0, 10)
plot_path_of(0, 42)
plot_path_of(0, 17)
plot_path_of(0, 35)
plot_path_of(0, 75)
plot_path_of(0, 12)


# In[ ]:


plot_path_of(1, 1)
plot_path_of(1, 13)
plot_path_of(1, 7)
plot_path_of(1, 5)
plot_path_of(1, 10)
plot_path_of(1, 42)
plot_path_of(1, 17)
plot_path_of(1, 35)
plot_path_of(1, 75)
plot_path_of(1, 11)


# In[ ]:


plot_path_of(2, 1)
plot_path_of(2, 13)
plot_path_of(2, 7)
plot_path_of(2, 5)
plot_path_of(2, 10)
plot_path_of(2, 42)
plot_path_of(2, 17)
plot_path_of(2, 35)
plot_path_of(2, 75)
plot_path_of(2, 97)


# In[ ]:


plot_path_of(3, 1)
plot_path_of(3, 13)
plot_path_of(3, 7)
plot_path_of(3, 5)
plot_path_of(3, 10)
plot_path_of(3, 42)
plot_path_of(3, 17)
plot_path_of(3, 35)
plot_path_of(3, 75)
plot_path_of(3, 98)


# In[ ]:


plot_path_of(4, 1)
plot_path_of(4, 13)
plot_path_of(4, 7)
plot_path_of(4, 5)
plot_path_of(4, 10)
plot_path_of(4, 42)
plot_path_of(4, 17)
plot_path_of(4, 35)
plot_path_of(4, 75)
plot_path_of(4, 99)


# In[ ]:


plot_path_of(5, 1)
plot_path_of(5, 13)
plot_path_of(5, 7)
plot_path_of(5, 5)
plot_path_of(5, 10)
plot_path_of(5, 42)
plot_path_of(5, 17)
plot_path_of(5, 35)
plot_path_of(5, 75)
plot_path_of(5, 42)


# In[ ]:


plot_path_of(6, 1)
plot_path_of(6, 13)
plot_path_of(6, 7)
plot_path_of(6, 5)
plot_path_of(6, 10)
plot_path_of(6, 42)
plot_path_of(6, 17)
plot_path_of(6, 35)
plot_path_of(6, 75)
plot_path_of(6, 13)


# In[ ]:


plot_path_of(7, 1)
plot_path_of(7, 13)
plot_path_of(7, 7)
plot_path_of(7, 5)
plot_path_of(7, 10)
plot_path_of(7, 42)
plot_path_of(7, 17)
plot_path_of(7, 35)
plot_path_of(7, 75)
plot_path_of(7, 4)


# In[ ]:


plot_path_of(8, 1)
plot_path_of(8, 13)
plot_path_of(8, 7)
plot_path_of(8, 5)
plot_path_of(8, 10)
plot_path_of(8, 42)
plot_path_of(8, 17)
plot_path_of(8, 35)
plot_path_of(8, 75)
plot_path_of(8, 5)


# In[ ]:


plot_path_of(9, 1)
plot_path_of(9, 13)
plot_path_of(9, 7)
plot_path_of(9, 5)
plot_path_of(9, 10)
plot_path_of(9, 42)
plot_path_of(9, 17)
plot_path_of(9, 35)
plot_path_of(9, 75)
plot_path_of(9, 8)

