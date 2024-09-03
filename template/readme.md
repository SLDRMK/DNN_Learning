## Here is to design a easy-to-use deep learning framework.
### From SLDRMK

### Dataloader Design
1. The **root directory** of the data should be specified.
2. The **split of data** should be stored in a json file, defaultly named "split.json".
3. The **data** should be organized in the following levels:
   - The whole dataset is a dictionary, with keys as the splits.
   - Splits: train, val, test,... (Each split should be a list)
   - Single data: a dictionary of different values, including paths of data, class labels, etc.
   - All the paths should be relative to the root directory.
4. The **IO** functions should be implemented in `./data/io.py`, including in the `load_data`, `save_data` function, which would be processed by the dataloader.
5. The **Transfrom Classes** should be implemented in `./data/transforms.py`, called by the dataloader in the form of 
```python
transform = Compose([
   (transform1, params1),
   (transform2, params2),
  ...
])

data = transform(data, params)
```

### Model Design