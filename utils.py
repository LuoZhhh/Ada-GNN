import torch
import numpy as np


def save_best_model(model, path):
    torch.save(model.state_dict(), path)


def load_best_model(model, path):

    model.load_state_dict(torch.load(path))

    return model

def softmax(x):

    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    sm_x = exp_x/sum_exp_x

    return sm_x