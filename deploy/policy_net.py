import torch
import torch.nn as nn
import pickle
import numpy as np

def get_params(policy_file: str):
    with open(policy_file, 'rb') as f:
        params = pickle.load(f)
    if len(params)==3:
        mean=params[0].mean["state"]
        std=params[0].std["state"]
        param_dict = params[1]["params"]

        weights = []
        biases = []
        for layer_name in param_dict:
            weights.append(param_dict[layer_name]["kernel"])
            biases.append(param_dict[layer_name]["bias"])

        return mean,std,weights, biases
    else:
        mean=params[0].mean["state"]
        std=params[0].std["state"]
        param_dict = params[1].policy['params']
        weights = []
        biases = []
        # print(params[0].mean["state"])
        # print(param_dict["hidden_0"])
        for layer_name in param_dict:
            weights.append(param_dict[layer_name]["kernel"])
            biases.append(param_dict[layer_name]["bias"])

        return mean,std,weights, biases


class MLP(nn.Module):
    def __init__(self, weights, biases, activation_fn=nn.ReLU(),mean=None,std=None):
        super(MLP, self).__init__()
        self.mean = torch.tensor(np.asarray(mean), dtype=torch.float32, requires_grad=False)
        self.std = torch.tensor(np.asarray(std), dtype=torch.float32, requires_grad=False)

        self.layers = nn.ModuleList()
        self.activation_fn = activation_fn
        
        layer_sizes = [weights[0].shape[0]] 
        for w in weights:
            layer_sizes.append(w.shape[1])  

        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        
        self.load_params(weights, biases)

    def forward(self, x):
        x=(x-self.mean)/self.std
        for layer in self.layers[:-1]:  
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)  

        loc, _ = torch.chunk(x, 2, dim=-1)  

        return torch.tanh(loc)
    def load_params(self, weights, biases):
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            weight_tensor = torch.tensor(np.asarray(weight), dtype=torch.float32).T
            bias_tensor = torch.tensor(np.asarray(bias), dtype=torch.float32)
            
            self.layers[i].weight.data = weight_tensor
            self.layers[i].bias.data = bias_tensor

def policy_net(policy_file: str,activation_fn=nn.SiLU()):
    mean,std,weights, biases = get_params(policy_file)
    model = MLP(weights, 
                biases, 
                activation_fn,
                mean,
                std)
    return model

