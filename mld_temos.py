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

labels = read_json('deps/inference/labels.json')

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="sample_eval_latent")
def _calc_temos_score(cfg: DictConfig):
    return calc_temos_score(cfg)


def fix_config_if_needed(cfg):
    if 'gpt_path' not in cfg.data:
        cfg.data['gpt_path'] = '${path.deps}/gpt/gpt3-labels.json'


def load_temos(classifier_path, ckpt_name="last"):
    from hydra.utils import instantiate
    temos_path = Path(classifier_path)
    temoscfg = OmegaConf.load(temos_path / ".hydra/config.yaml")

    # Overload it
    logger.info("Loading Evaluation Classifier model")
  
    # Instantiate all modules specified in the configs
    temos_model = instantiate(temoscfg.model,
                              nfeats=135,
                              logger_name="none",
                              nvids_to_save=None,
                              _recursive_=False)
    if ckpt_name == 'last':
        last_ckpt_path = temos_path / "checkpoints/last.ckpt"
    else:
        last_ckpt_path = temos_path / f"checkpoints/latest-epoch={ckpt_name}.ckpt"
    from collections import OrderedDict
    state_dict = torch.load('/is/cluster/fast/nathanasiou/logs/sinc/sinc-arxiv/temos-bs64x1-scheduler/babel-amass/checkpoints/latest-epoch=599.ckpt', map_location='cpu')['state_dict']

    model_dict = OrderedDict()
    for k,v in state_dict.items():
        if k.split('.')[0] not in ['eval_model', 'metrics']:
            model_dict[k] = v
    temos_model.load_state_dict(model_dict, strict=True)

    # Load the last checkpoint
    # temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
    temos_model.eval()

    return temos_model, temoscfg


def get_metric_paths(sample_path: Path, set: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    if set == 'pairs':
        metric_str = "babel_metrics_sp"
    else:
        metric_str = "babel_metrics_all"

    if onesample:
        file_path = f"{fact_str}{metric_str}_{split}{extra_str}"
        save_path = sample_path / file_path
        return save_path
    else:
        file_path = f"{fact_str}{metric_str}_{split}_multi"
        avg_path = sample_path / (file_path + "_avg")
        best_path = sample_path / (file_path + "_best")
        return avg_path, best_path


def calc_temos_score(newcfg: DictConfig) -> None:
    # Load last config

    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")

    fix_config_if_needed(prevcfg)

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    onesample = cfg_mean_nsamples_resolution(cfg)
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    logger.info("Sample script. The outputs will be stored in:")

    path = output_dir / 'metrics'
    path.mkdir(exist_ok=True, parents=True)
    file_path = f"TemosScore_{cfg.set}_{cfg.naive}_{cfg.ckpt_name}"
    metrics_path = path / file_path

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
    state_dict = torch.load('/is/cluster/fast/nathanasiou/logs/sinc/sinc-arxiv/temos-bs64x1-scheduler/babel-amass/checkpoints/latest-epoch=599.ckpt', map_location='cpu')
    # extract encoder/decoder
    from collections import OrderedDict
    decoder_dict = OrderedDict()
    encoder_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        if k.split(".")[0] == "motionencoder":
            name = k.replace("motionencoder.", "")
            encoder_dict[name] = v
        if k.split(".")[0] == "motiondecoder":
            name = k.replace("motiondecoder.", "")
            decoder_dict[name] = v

    model.vae_encoder.load_state_dict(encoder_dict, strict=True)
    model.vae_decoder.load_state_dict(decoder_dict, strict=True)


    logger.info(f"Model '{cfg.model.modelname}' loaded")

    # Load the last checkpoint
    model = model.load_from_checkpoint(last_ckpt_path)
    model.eval()
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
    # full_dict = {}
    temos_model, _ = load_temos(cfg.classifier_path, cfg.classifier_ckpt)

    metrics = []
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
                try:
                    if model.hparams.gpt_proxy:
                        gpt_parts = batch['bp-gpt']
                    else:
                        gpt_parts = None
                except AttributeError:
                    gpt_parts = None

                dtype_sample = keyid.split('-')[0]
                is_sp = dtype_sample == 'spatial_pairs'
                if is_sp and cfg.naive == "gpt":
                    from sinc.tools.frank import combine_motions
                    gpt_parts = batch['bp-gpt'][0]

                    motion1 = model.text_to_motion_forward([[cur_texts[0][0]]],
                                                           cur_lens,
                                                           gpt_parts=None,
                                                           return_motion="rotsd")

                    motion2 = model.text_to_motion_forward([[cur_texts[0][1]]],
                                                           cur_lens,
                                                           gpt_parts=None,
                                                           return_motion="rotsd")


                    # rots and transl
                    frank_motion = combine_motions(motion1, motion2, gpt_parts[0], gpt_parts[1], squeeze=True)
                    frank_datastruct = model.Datastruct(rots_=frank_motion)

                    motion = model.motion_from_datastruct(frank_datastruct, return_type=motion_type)

                    # just in case
                    motion1 = model.motion_from_datastruct(model.Datastruct(rots_=motion1), return_type=motion_type)
                    motion2 = model.motion_from_datastruct(model.Datastruct(rots_=motion2), return_type=motion_type)
                elif is_sp and cfg.naive == "concat":
                    # concat_text = [[" while ".join(cur_texts[0])]]
                    motion = model.text_to_motion_forward(cur_texts,
                                                          cur_lens,
                                                          gpt_parts=gpt_parts,
                                                          return_motion=motion_type)

                else:
                    motion = model(cur_texts,cur_lens)
                # motion = datastruct.rots
                # rots, transl = motion.rots, motion.trans

                # from sinc.transforms.smpl import RotTransDatastruct
                # final_datastruct = self.Datastruct(
                # rots_=RotTransDatastruct(rots=rots, trans=transl))

                distribution_ref = temos_model.motionencoder(torch.squeeze(temos_model.transforms.rots2rfeats(one_data["datastruct"]))[None])
                distribution_motion = temos_model.motionencoder(torch.squeeze(temos_model.transforms.rots2rfeats(motion))[None]) 

                mu_ref = distribution_ref.mean[0,0]
                mu_motion = distribution_motion.mean[0,0]

                # dist = torch.linalg.norm(mu_motion-mu_ref)
                metric = 2*(1-torch.nn.CosineSimilarity()(mu_motion[None], mu_ref[None]))
                metrics.append(metric.detach().cpu().numpy())

    metrics_dict = dict(zip(keyids, metrics))
    metrics_dict = {key: round(1-val.item()/4, 5) for key, val in metrics_dict.items()}

    mu_cos_metric = float(np.mean(metrics))
    metrics_dict['TOTAL'] = 1-mu_cos_metric/4
    metrics_dict['info'] = f'naive={cfg.naive}__set={cfg.set}__ckpt={cfg.ckpt_name}'
    
    #write_json(metrics_dict, metrics_path)    
    logger.info(f"This metric is for the model which is under: {str(last_ckpt_path)}")
    logger.info(f"The cosine similarity loss is {metrics_dict['TOTAL']}.")
    logger.info(f"This saved file is: {metrics_path}")
    return metrics_dict

if __name__ == '__main__':
    _calc_temos_score()
