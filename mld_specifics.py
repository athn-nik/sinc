import importlib

import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
from hydra.utils import to_absolute_path, get_original_cwd

def get_module_config(cfg_model, path="modules"):
    files = os.listdir(f"{get_original_cwd()}/configs_mld/{path}/")
    for file in files:
        if file.endswith('.yaml'):
            with open(f"{get_original_cwd()}/configs_mld/{path}/" + file, 'r') as f:
                cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):

    # update config from files
    cfg_base = OmegaConf.load(f'{get_original_cwd()}/configs_mld/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(f"{get_original_cwd()}/configs_mld/config_mld_humanml3d.yaml"))
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(f"{get_original_cwd()}/configs_mld/assets.yaml")
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    return cfg

def get_model(cfg, datamodule, phase="train"):
    modeltype = cfg.model.model_type
    if modeltype == "mld":
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="mld.models")
    import ipdb; ipdb.set_trace()
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    return Model(cfg=cfg, datamodule=datamodule)
