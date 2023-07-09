import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sinc.data.tools.collate import collate_length_and_text
import sinc.launch.prepare
# from sinc.render.mesh_viz import visualize_meshes
# from sinc.render.video import save_video_samples, stack_vids
import torch
from sinc.transforms.base import Datastruct
from sinc.utils.inference import cfg_mean_nsamples_resolution, get_path
from sinc.utils.file_io import read_json, write_json
from sinc.transforms.smpl import RotTransDatastruct
from sinc.transforms.rots2joints.smplh import SMPLH
labels = read_json('deps/inference/labels.json')

from sinc.render.mesh_viz import visualize_meshes
from sinc.render.video import save_video_samples

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="mld_demo")
def _calc_temos_score(cfg: DictConfig):
    return calc_temos_score(cfg)


def fix_config_if_needed(cfg):
    if 'gpt_path' not in cfg.data:
        cfg.data['gpt_path'] = '${path.deps}/gpt/gpt3-labels.json'


def calc_temos_score(newcfg: DictConfig) -> None:
    # Load last config

    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")

    fix_config_if_needed(prevcfg)
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    SMPL_layer = SMPLH(path=f'{cfg.path.data}/smpl_models/smplh', jointstype='vertices', gender='male')
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    logger.info("Sample script. The outputs will be stored in:")


    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)

    pl.seed_everything(cfg.seed)
    logger.info("Loading data module")
    # only pair evaluation to be fair
    # keep same order

    cfg.data.dtype = 'spatial_pairs+seg+seq'

    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    dataset = getattr(data_module, f"{cfg.split}_dataset")

    from tqdm import tqdm

    logger.info("Loading model")
    # Instantiate all modules specified in the configs
    from mld_specifics import parse_args
    cfg_for_mld = parse_args()  # parse config file

    # MLD specific changes
    from sinc.model.mld import MLD
    model = MLD(cfg_for_mld, cfg.transforms, cfg.path)
    # state_dict = torch.load('/is/cluster/fast/nathanasiou/logs/sinc/sinc-arxiv/temos-bs64x1-scheduler/babel-amass/checkpoints/latest-epoch=599.ckpt', map_location='cpu')
    # # extract encoder/decoder
    # from collections import OrderedDict
    # decoder_dict = OrderedDict()
    # encoder_dict = OrderedDict()
    # for k, v in state_dict['state_dict'].items():
    #     if k.split(".")[0] == "motionencoder":
    #         name = k.replace("motionencoder.", "")
    #         encoder_dict[name] = v
    #     if k.split(".")[0] == "motiondecoder":
    #         name = k.replace("motiondecoder.", "")
    #         decoder_dict[name] = v

    # model.vae_encoder.load_state_dict(encoder_dict, strict=True)
    # model.vae_decoder.load_state_dict(decoder_dict, strict=True)


    logger.info(f"Model '{cfg.model.modelname}' loaded")
    state_dict = torch.load(last_ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)

    # Load the last checkpoint
    # temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
    model.eval()

    # Load the last checkpoint
    # model = model.load_from_checkpoint(last_ckpt_path)
    # model.eval()
    
    
    logger.info("Model weights restored")
    model.sample_mean = cfg.mean
    model.fact = cfg.fact


    # trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))

    logger.info("Trainer initialized")

    model.transforms.rots2joints.jointstype = cfg.jointstype

    # ds = model.transforms.Datastruct
    if cfg.set == 'submission':
        from sinc.utils.inference import sinc_eval_set
        keyids = sinc_eval_set
    elif cfg.set == 'small':
        from sinc.utils.inference import validation_nostand_notrain
        keyids = validation_nostand_notrain
    elif cfg.set == 'supmat':
        from sinc.utils.inference import sinc_supmat
        keyids = sinc_supmat
    else:
        if cfg.set == 'pairs':
            keyids = [k for k in dataset.keyids if k.split('-')[0] == 'spatial_pairs']
        elif cfg.set == 'single':
            keyids = [k for k in dataset.keyids if k.split('-')[0] in ['seq', 'seg']]
        else:
            keyids = dataset.keyids


    motion_type = "rotsd"
    outd = Path(cfg.savedir)
    outd.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        with tqdm(total=len(keyids), position=0, leave=True) as pbar:

            for keyid in (pbar := tqdm(keyids,  position=0, leave=True)):
                
                pbar.set_description(f"Processing {keyid}")            
                one_data = dataset.load_keyid(keyid, mode='inference')
                from sinc.data.tools import collate_text_and_length
                batch = collate_text_and_length([one_data])
                               
                cur_lens = batch['length']
                cur_texts = [list(batch['text'][0])]

                # batch_size = 1 for reproductability
                # fix the seed
                pl.seed_everything(0)
                dtype_sample = keyid.split('-')[0]
                is_sp = dtype_sample == 'spatial_pairs'
                
                motion = model(cur_texts,cur_lens)
                # motion = datastruct.rots
                # rots, transl = motion.rots, motion.trans

                # from sinc.transforms.smpl import RotTransDatastruct
                # final_datastruct = self.Datastruct(
                # rots_=RotTransDatastruct(rots=rots, trans=transl))
                ds_ = RotTransDatastruct(rots=motion.rots, trans=motion.trans)

                motion_verts = SMPL_layer(ds_).numpy()
                vid_ = visualize_meshes(motion_verts.squeeze())
                vid_p = save_video_samples(vid_,
                            f'{str(outd)}/{keyid}.mp4',
                            cur_texts[0],
                            fps=30)


    logger.info(f"The samples are saved under: {outd}")
    return 

if __name__ == '__main__':
    _calc_temos_score()
