from G2M_model import Graph2Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import Glucose_sim_Config3
def main(adj_matrix, Ux: list):
    new_model = Graph2Model(adj_matrix, Ux)
    new_model.load_state_dict(torch.load("./ckpt/glucose_model_change_isu.pth"))

