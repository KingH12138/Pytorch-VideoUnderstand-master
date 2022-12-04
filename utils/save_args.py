import yaml
import argparse


def save(args,save_path):
    """
    :param args:
    >> parser = argparse.ArgumentParser()
    >> args = parser.parse_args()
    """
    args_dict = vars(args)  # object->dict
    with open(save_path, 'w') as f:
        obj = yaml.dump(args_dict)
        f.write(obj)


def load(save_path):
    with open(save_path,'r') as f:
        content = yaml.safe_load(f.read())
    args = argparse.Namespace(**content)
    return args
