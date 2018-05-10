"""
Helpers for evaluating models.
"""

from .reptile import Reptile
from .variables import weight_decay
import numpy as np

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    total_correct = 0
    acc_list = []
    for _ in range(num_samples):
        this_correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                          model.minimize_op, model.predictions,
                                          num_classes=num_classes, num_shots=num_shots,
                                          inner_batch_size=eval_inner_batch_size,
                                          inner_iters=eval_inner_iters, replacement=replacement)
        total_correct += this_correct
        acc_list.append(this_correct/num_classes)
    means = np.mean(acc_list, 0)
    stds = np.std(acc_list, 0)
    ci95 = 1.96*stds/np.sqrt(num_samples)
    print(means, stds, ci95)
    return total_correct / (num_samples * num_classes)
