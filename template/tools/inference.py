from data.io import load_data, save_data

def inference(model, input_data_path, device='cuda:0', save_path=None):
    input_data = load_data(input_data_path)
    output = model(input_data.to(device)).detach().cpu().numpy()
    if save_path is not None:
        save_data(output, save_path)
    return output