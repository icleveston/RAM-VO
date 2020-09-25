import time
import os

# 2019041500 - use this tf nightly version
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

from sklearn.model_selection import train_test_split

from model.ram import RecurrentAttentionModel

from data.augmented_mnist import minibatcher
from data.augmented_mnist import get_mnist

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

batch_size = 100

(X_train, y_train),(X_test, y_test) = get_mnist(True, True, False)

print(X_train.shape, y_train.shape, np.max(X_train), np.min(X_train))
print(X_test.shape, y_test.shape, np.max(X_test), np.min(X_test))

def objective_function(learning_rate, std):
    learning_rate = np.round(learning_rate, 8).astype(np.float32)
    std = np.round(std, 2).astype(np.float32)
    print("params:", learning_rate, std)
    
    ram = RecurrentAttentionModel(time_steps=7,
                                  n_glimpses=1, 
                                  glimpse_size=8,
                                  num_classes=10,
                                  max_gradient_norm=1.0,
                                  std=std)
    adam_opt = tf.keras.optimizers.Adam(learning_rate)
    
    for timestep in range(50):
        # training step
        accuracy = []
        batcher = minibatcher(X_train, y_train, batch_size, True)
        for X, y in batcher:
            with tf.GradientTape() as tape:
                logits = ram(X)
                loss, _, reward, _ = ram.hybrid_loss(logits, y)
                accuracy.append(reward)
                # calculate gradient and do gradient descent
                gradients = tape.gradient(loss, ram.trainable_variables)
                adam_opt.apply_gradients(zip(gradients, ram.trainable_variables))
    return np.mean(accuracy)

# Bounded region of parameter space
pbounds = {'learning_rate': (1e-8, 0.1), 'std': (0, 1)}

# optimizer
bayes_opt = BayesianOptimization(f=objective_function,
                                 pbounds=pbounds,
                                 random_state=42)

if not os.path.isfile("logs/mnist.json"):
    logger = JSONLogger(path="logs/mnist.json")
    bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
else:
    load_logs(bayes_opt, logs=["logs/mnist.json"])
    logger = JSONLogger(path="logs/mnist.json")
    bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
    
print("subscribers:", bayes_opt.get_subscribers(Events.OPTMIZATION_STEP))
print("points:", len(bayes_opt.space))

bayes_opt.maximize(init_points=5, # how much the optimizer explores
                   n_iter=2000,
                   kappa=2.5, # kernel parameter
                   xi=0.0)
print(bayes_opt.max)
