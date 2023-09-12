import yaml

def load_config(model_name='swin'):
    if model_name == 'swin':
        f_name = "config/swin_config.yaml"
    elif model_name == 'resnet':
        f_name = "config/resnet_config.yaml"
    with open(f_name,"r") as f:
        c = f.read()
    config = yaml.safe_load(c)
    return config

if __name__ == "__main__":
    print(load_config())