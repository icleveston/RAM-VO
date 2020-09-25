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
from data.augmented_mnist import get_translated_mnist

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

batch_size = 100

def create_translated_mnist():
    (X_train, y_train),(X_test, y_test) = get_translated_mnist(60, 60)

    X_train = X_train.reshape(-1, 60, 60, 1)
    X_test = X_test.reshape(-1, 60, 60, 1)

    X_train = (X_train/255).astype(np.float32)
    X_test = (X_test/255).astype(np.float32)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (X_train, y_train),(X_test, y_test)

#(X_train, y_train),(X_test, y_test) = create_translated_mnist()

#print(X_train.shape, y_train.shape, np.max(X_train), np.min(X_train), X_train.dtype)
#print(X_test.shape, y_test.shape, np.max(X_test), np.min(X_test), X_test.dtype)

def objective_function(learning_rate, std):
    learning_rate = np.round(learning_rate, 8).astype(np.float32)
    std = np.round(std, 2).astype(np.float32)
    print("params:", learning_rate, std)
    
    ram = RecurrentAttentionModel(time_steps=8,
                                  n_glimpses=3, 
                                  glimpse_size=12,
                                  num_classes=10,
                                  max_gradient_norm=1.0,
                                  std=std)
    adam_opt = tf.keras.optimizers.Adam(learning_rate)
    
    for timestep in range(50):
        # training step
        (X_train, y_train),(X_test, y_test) = create_translated_mnist()
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
                                 random_state=42,
                                 verbose=2)

if not os.path.isfile("logs/translated-mnist.json"):
    logger = JSONLogger(path="logs/translated-mnist.json")
    bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
else:
    load_logs(bayes_opt, logs=["logs/translated-mnist.json"])
    logger = JSONLogger(path="logs/translated-mnist.json")
    bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
print("loaded points:", len(bayes_opt.space))

bayes_opt.maximize(init_points=2, # how much the optimizer explores
                   n_iter=2000,
                   kappa=2.5, # kernel parameter
                   xi=0.0)
