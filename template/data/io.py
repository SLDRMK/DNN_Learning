import numpy as np
import PIL.Image as Image

def load_data(path):
    if path.endswith('.npy'):
        return load_npy(path)
    elif path.endswith('.txt'):
        return load_txt(path)
    elif path.endswith(('.jpg', '.jpeg', '.png')):
        return load_img(path)
    else:
        raise ValueError('Unsupported file format')

def save_data(data, path):
    if path.endswith('.npy'):
        save_npy(data, path)
    elif path.endswith('.txt'):
        save_txt(data, path)
    elif path.endswith(('.jpg', '.jpeg', '.png')):
        save_img(data, path)
    else:
        raise ValueError('Unsupported file format')

def load_img(path):
    img = Image.open(path)
    return np.array(img)

def save_img(data, path):
    img = Image.fromarray(data)
    img.save(path)

def load_npy(path):
    return np.load(path)

def save_npy(data, path):
    np.save(path, data)


def load_txt(path):
    with open(path, 'r') as f:
        return f.read()

def save_txt(data, path):
    with open(path, 'w') as f:
        f.write(data)