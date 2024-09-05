import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def write_config(config_file, config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    config_file = './template/configs/default_config.json'
    
    config = {
        "model": {
            "name": "ResNet",
            "block" : "BasicBlock",
            "num_classes": 10,
            "num_blocks" : [3, 4, 6, 3],
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0005,
        },
        "scheduler": {
            "name": "StepLR",
            "step_size": 7,
            "gamma": 0.1,
        },
        "data": {
            "name": "CIFAR10",
            "root": "./datasets/DefaultDataset",
            "split_file": "split.json",
            "batch_size": 128,
            "num_workers": 4,
            "transforms": [{
                "name" : "Normalize", 
                "params" : {"target": "input"}, 
            }, {
                "name" : "ToTensor", 
                "params" : {"target_keys" : ["input", "GT"]}
            }],
        }
    }

    write_config(config_file, config)

    loaded_config = load_config(config_file)
    print(loaded_config)