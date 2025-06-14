import torch
import torch.utils.data as data 
import numpy as np
import pickle



def metropolis_algorithm(f, initial_state, num_samples, burn_in):
    chain = []
    current_state = initial_state
    for _ in range(num_samples + burn_in):
        proposed_state = np.random.uniform(-6, 6)
        acceptance_ratio = min(1, f(proposed_state) / f(current_state))
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        chain.append(current_state)
    return chain[burn_in:]

def f(x):
    return 1 / (1 + x**2)#np.exp(-x**2) + np.exp(-(x-2)**2)#1 / (1 + x**2)  # Example: Gaussian distribution


def generate_mixture_of_gaussians(num_of_points):
    n = num_of_points // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(num_of_points-n,))
    return np.concatenate([gaussian1, gaussian2])

class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

n_train, n_test = 50000, 40000
train_data = metropolis_algorithm(f, 1, n_train, 2000)
test_data = metropolis_algorithm(f, 1, n_test, 2000)
# Save to file
with open('my_array.pkl', 'wb') as h:
    pickle.dump(train_data, h)

train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=128, shuffle=True)
test_loader = data.DataLoader(NumpyDataset(test_data), batch_size=128, shuffle=True)