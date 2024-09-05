def build_model(config:dict):
    if config["model"]["name"] == "ResNet":
        from models.ResNet import BasicBlock, ResNet
        if config["model"]["block"] == "BasicBlock":
            block = BasicBlock
        
        num_blocks = config["model"]["num_blocks"]
        num_classes = config["model"]["num_classes"]

        return ResNet(
            block=block,
            num_blocks=num_blocks,
            num_classes=num_classes
        )
    else:
        raise ValueError("Invalid model name")
    
def build_optimizer(config:dict, model):
    if config["optimizer"]["name"] == "SGD":
        from torch.optim import SGD
        optimizer = SGD(
            params=model.parameters(),
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
            weight_decay=config["optimizer"]["weight_decay"]
        )
    else:
        raise ValueError("Invalid optimizer name")
    
    return optimizer

def build_scheduler(config:dict, optimizer):
    if config["scheduler"]["name"] == "StepLR":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"]
        )
    else:
        raise ValueError("Invalid scheduler name")
    
    return scheduler

def build_transforms(config:dict):
    transform_dicts = config["transforms"]
    transforms = []
    for transform_dict in transform_dicts:
        params = transform_dict["params"]
        if transform_dict["name"] == "Normalize":
            from data.transforms import Normalize
            transform = Normalize()
        elif transform_dict["name"] == "ToTensor":
            from data.transforms import ToTensor
            transform = ToTensor()
        else:
            raise ValueError("Invalid transform name")
        
        transforms.append((transform, params))
    from data.transforms import Compose
    transforms = Compose(transforms)
    return transforms

def build_dataloader(config:dict, split:str, shuffle:bool=True):
    transforms = build_transforms(config)
    if config["dataset"]["name"] == "CIFAR10":
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        from torch.utils.data import DataLoader
        transform = ToTensor()
        dataset = CIFAR10(
            root=config["dataset"]["root"],
            train=split=="train",
            download=True,
            transform=transform
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=shuffle,
            num_workers=config["dataloader"]["num_workers"]
        )
    elif config["dataset"]["name"] == "DefaultDataset":
        from data.defalt_dataset import DefaultDataset
        dataset = DefaultDataset(
            root=config["dataset"]["root"],
            split=split,
            transform=config["dataset"]["transform"]
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=shuffle,
            num_workers=config["dataloader"]["num_workers"]
        )
    else:
        raise ValueError("Invalid dataset name")
    
    return dataloader