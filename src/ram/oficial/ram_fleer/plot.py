import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np

def load_single(filename):
    """
    loads and returns a single experiment stored in filename
    returns None if file does not exist
    """
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        result = json.load(f)
    return result


# This is not a nice way to implement the different configuration scripts...
if len(sys.argv) > 1:
    file = load_single(sys.argv[1])
    if file is None:
        print "Wrong file name!"
        sys.exit(0)
else:
    print "Give Results-File as additional argument! \n " \
          "E.g. python plot.py ./results.json"
    sys.exit(0)

style = {
    "linewidth": 2, "alpha": .7, "linestyle": "-", "markersize": 7}


#x = np.arange(len(file['accuracy']))
x = np.asarray(file['learning_steps'])
y_mean = np.asarray(file['accuracy'])
y_sem = np.asarray(file['accuracy_std'])
y_mean *= 100.
y_sem *= 100.
fig = plt.figure()
plt.plot(x, y_mean, **style)

min_ = np.inf
max_ = - np.inf
plt.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=.3)
max_ = max(np.max(y_mean + y_sem), max_)
min_ = min(np.min(y_mean - y_sem), min_)
# adjust visible space
y_lim = [min_ - .1 * abs(max_ - min_), max_ + .1 * abs(max_ - min_)]
if min_ != max_:
    plt.ylim(y_lim)

plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Accuracy [%]", fontsize=16)
plt.grid(True)

plt.show()

