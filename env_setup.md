### Environment Setup
*Note: This is a configuration explanation file for the setup of the pytorch virtual environment of 40 series NVIDIA GPUs.*

1. Install Anaconda ```64-Bit (x86) Installer``` from the official website: https://www.anaconda.com/download/success. \
Run `conda init` to initialize anaconda.

2. Create a new environment with the name `#env_name#` using the following command:
```
conda create -n #env_name# python=3.8
```

3. Activate the environment using the following command:
```
conda activate #env_name#
```

4. Install CUDA Toolkit 11.8 using the following command:
```
conda install cudatoolkit=11.8
```

5. Install PyTorch using the following command:
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Other options can be checked at https://pytorch.org/get-started/previous-versions/.

6. PyTorch Tutorials can be found at
```
https://pytorch.org/tutorials/.
```
Introduction to PyTorch can be found at
```
https://pytorch.org/tutorials/beginner/basics/intro.html
```