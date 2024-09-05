import torch
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for (transform, params) in self.transforms:
            data = transform(data, params)
        return data
    
class ToTensor(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data, params):
        for key in params["target_keys"]:
            data[key] = torch.from_numpy(data[key])
        return data
    
class Normalize(object):
    def __init__(self):
        pass
        
    def __call__(self, data, params=None):
        if 'mean' not in params or 'std' not in params:
            target_key = params['target']
            params['mean'] = np.mean(data[target_key], axis=0)
            params['std'] = np.std(data[target_key], axis=0)
        for key in params['keys']:
            data = (data - params['mean']) / params['std']
        return data
