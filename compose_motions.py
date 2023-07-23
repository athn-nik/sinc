from os import name
from typing import Optional, List, Dict
import logging
import joblib
import numpy as np
from sinc.render.mesh_viz import visualize_meshes
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from sinc.utils.file_io import read_json
from sinc.data.tools.extract_pairs import extract_frame_labels

from sinc.tools.frank import text_to_bp, text_list_to_bp

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]

def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel_v2.1/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel


def load_gpt_labels(path: str):
    if 'gpt/gpt3-labels-list.json' not in str(path):
        path = path.replace('gpt/gpt3-labels.json','gpt/gpt3-labels-list.json')
    gpt_labels = read_json(path)
    return gpt_labels

def collate_rots_and_text(lst_elements: List) -> Dict:
    # collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    batch = {# Collate with padding for the datastruct
             "rots": 
             collate_tensor_with_padding([el["rots"] for el in lst_elements]),
             "rots_a": 
             collate_tensor_with_padding([el["rots_a"] for el in lst_elements]),
             "rots_b": 
             collate_tensor_with_padding([el["rots_b"] for el in lst_elements]),
             "trans_a": 
             collate_tensor_with_padding([el["trans_a"] for el in lst_elements]),
             "trans_b": 
             collate_tensor_with_padding([el["trans_b"] for el in lst_elements]),

             # Collate normally for the length
            #  "datastruct_a": [x["datastruct_a"] for x in lst_elements],
            #  "datastruct_b": [x["datastruct_b"] for x in lst_elements],
             "length": [x["length"] for x in lst_elements],
             # Collate the text
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch

class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self,
                 datapath: str,
                 split: str = "train",
                 sampler=None,
                 progress_bar: bool = True,
                 load_with_rot=True,
                 downsample=True,
                 tiny: bool = False,
                 synthetic: Optional[bool] = False,
                 proportion_synthetic: float = 1.0,
                 centered_compositions: Optional[bool] = False,
                 dtype: str = 'seg+seq',
                 mode: str = 'train',
                 simultaneous_max: int = 2,
                 **kwargs):
        self.simultaneous_max = simultaneous_max
        self.split = split
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.dtype = dtype  # seg or seq or empty string for segments or sequences
        # or all of the stuff --> 'seg', 'seq', 'pairs', 'pairs_only', ''
        self.synthetic = synthetic
        
        self.proportion_synthetic = proportion_synthetic
        self.centered_compositions = centered_compositions
 
        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        super().__init__()
        if tiny:
            data_for_split = get_split(path=datapath,
                                       split=split,
                                       subset='_tiny')
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        else:
            data_for_split = get_split(path=datapath, split=split)
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        gpt_path = 'deps/gpt/gpt3-labels-list.json'

        gpt_labels_full_sent = load_gpt_labels(gpt_path)

        fname2key = get_babel_keys(path=datapath)
        motion_data_rots = {}
        motion_data_trans = {}

        joints_data = {}
        texts_data = {}
        durations = {}
        dtypes = {}
        gpt_labels = {}
        if progress_bar:
            enumerator = enumerate(
                tqdm(data_for_split, f"Loading BABEL {split}"))
        else:
            enumerator = enumerate(data_for_split)

        if tiny:
            maxdata = 1000
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        valid_data_len = 0
        all_data_len = 0
        # discard = read_json('/home/nathanasiou/Desktop/conditional_action_gen/teach/data/amass/amass_cleanup_seqs/BMLrub.json')
        req_dtypes = self.dtype.split('+')
        logger.info(f'The required datatypes are : {req_dtypes}')

            
        for i, sample in enumerator:
            # if sample['fname'] in discard:
            #     num_bad_bml += 1
            #     continue

            # from temos.data.sampling import subsample
            all_data_len += 1
            # smpl_data = {x: smpl_data[x] for x in smpl_data.files}
            nframes_total = len(sample['poses'])
            last_framerate = sample['fps']
            babel_id = sample['babel_id']
            frames = np.arange(nframes_total)
            all_actions = extract_frame_labels(self.babel_annots[babel_id],
                                               fps=last_framerate,
                                               seqlen=nframes_total,
                                               max_simultaneous=self.simultaneous_max
                                               )
            # if not valid:
            #     invalid += 1
            #     continue

            possible_actions = {
                k: v
                for k, v in all_actions.items() if k in req_dtypes
            }

            for dtype, extracted_actions in possible_actions.items():

                for index, seg in enumerate(extracted_actions):
                    if isinstance(seg[-1], str) and seg[-1] == '':
                        # 1 bad label
                        continue
                    if dtype == 'separate_pairs':
                        frames = np.arange(seg[0][0], seg[-1][1])
                        duration = [(e - s + 1) for s, e in seg]
                        duration[-1] -= 1
                        if len(duration) == 2: duration.insert(1, 0)
                    if dtype == 'spatial_pairs':
                        frames = np.arange(seg[0], seg[1])
                        duration = seg[1] - seg[0]
                    else:
                        frames = np.arange(seg[0], seg[1])
                        duration = len(frames)
                    if duration < 20 or duration > 300:
                        continue
                    keyid = f'{dtype}-{babel_id}-{index}'
                    if self.dtype == 'separate_pairs':
                        texts_data[keyid] = seg[-1]
                        durations[keyid] = duration
                        # assert features.shape[0] == sum(duration), f' \nMismatch: {babel_id}, \n {seg_ids} \n {seg} \n {frames} \n {fpair} \n {duration}--> {features.shape[0]}  {sum(duration)}'
                    else:
                        if isinstance(seg[-1], tuple):
                            texts_data[keyid] = seg[-1]
                        else:
                            texts_data[keyid] = (
                                seg[-1], )
                        durations[keyid] = duration

                    gpt_labels[keyid] = []
                    try:
                        for a_t in texts_data[keyid]:
                            bp_list = text_list_to_bp(a_t, gpt_labels_full_sent)
                            # bp_list = text_to_bp(a_t, gpt_labels_full_sent)
                            gpt_labels[keyid].append(bp_list)
                    except:
                        import ipdb; ipdb.set_trace()
                    
                    smpl_data = {
                        "poses":
                        torch.from_numpy(sample['poses'][frames]).float(),
                        "trans":
                        torch.from_numpy(sample['trans'][frames]).float()
                    }
                    # pose: [T, 22, 3, 3]

                    valid_data_len += 1
                    from sinc.data.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data,
                                                              nohands=True)
                    
                    motion_data_rots[keyid] = smpl_data.rots
                    motion_data_trans[keyid] = smpl_data.trans

                    dtypes[keyid] = dtype
        if synthetic:
            # from a keyid, prepare what keyids is possible to be chosen
            from sinc.info.joints import get_compat_matrix
            self.compat_matrix = get_compat_matrix(gpt_labels_full_sent)

            from collections import defaultdict
            self.keyids_from_text = defaultdict(list)

            self.compat_seqs = {}
            for key, val in texts_data.items():
                if "seq" not in key and "seg" not in key:
                    continue
                self.keyids_from_text[val[0]].append(key)

            for key, val in texts_data.items():
                self.compat_seqs[key] = [
                    y for x in self.compat_matrix[val[0]]
                    for y in self.keyids_from_text[x]
                ]
        self.motion_data_trans = motion_data_trans
        self.motion_data_rots = motion_data_rots

        self.texts_data = texts_data
        self._split_index = list(motion_data_rots.keys())
        self._num_frames_in_sequence = durations
        self.keyids = list(self.motion_data_rots.keys())
        self.single_keyids = [x for x in self.keyids if "spatial_pairs" not in x]
        self.pairs_keyids = [x for x in self.keyids if "spatial_pairs" in x]
        self.sample_dtype = dtypes
        self.gpt_labels = gpt_labels

        if split == 'test' or split == 'val' or mode == 'inference':
            # does not matter should be removed, just for code to not break
            self.nfeats = 135  # len(self[0]["datastruct"].features[0])
        elif self.dtype == 'separate_pairs':
            self.nfeats = 135  # len(self[0]["features_0"][0])
        else:
            self.nfeats = 135 #self[0]["datastruct"].features.shape[-1]


    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences


    def load_keyid(self, keyid, mode='train', proportion=None):
        proportion = proportion if proportion is not None else self.proportion_synthetic

        # force composition in loading
        force_comp = False
        if "synth" in keyid:
            keyid, keyid_p = keyid.split("synth_")[1].split("_")
            force_comp = True
        text = self._load_text(keyid)

        check_compat = self.synthetic and self.sample_dtype[keyid] in ['seg', 'seq'] and (self.compat_seqs[keyid])
        if np.random.uniform() < proportion and check_compat:
            # GPT CASE
            # check compatibility with synthetic
            # randomly do synthetic data or not
            # if there is some compatibility
            # take a random pair
            keyid_p = np.random.choice(self.single_keyids)
            keyid_p = np.random.choice(self.compat_seqs[keyid])

            text_p = self._load_text(keyid_p)
            rots_a = self.motion_data_rots[keyid]
            trans_a = self.motion_data_trans[keyid]

            rots_b = self.motion_data_rots[keyid_p]
            trans_b = self.motion_data_trans[keyid_p]

            bp = self.gpt_labels[keyid][0]
            bp_p = self.gpt_labels[keyid_p][0]

    

            element = { 'rots': self.motion_data_rots[keyid],
                        'trans': self.motion_data_trans[keyid],
                        'length': [len(rots_a), len(rots_b)], 
                        'text': (text[0], text_p[0]),
                        'keyid': f"synth_{keyid}_{keyid_p}",
                        'bp-gpt': [bp, bp_p],
                        'rots_a': rots_a,
                        'trans_a': trans_a,
                        'rots_b': rots_b,
                        'trans_b': trans_b,
                        }

        else:

            element = {'rots': self.motion_data_rots[keyid],
                        'trans': self.motion_data_trans[keyid], 
                        'length': [len(self.motion_data_rots[keyid]), 0],
                        'text': text,
                        'keyid': keyid,
                        'bp-gpt': self.gpt_labels[keyid],
                        'rots_a':self.motion_data_rots[keyid],
                        'trans_a': self.motion_data_trans[keyid],
                        'rots_b': self.motion_data_rots[keyid],
                        'trans_b': self.motion_data_trans[keyid],
                        }
        return element


    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"



def gpt_compose(rots1, trans1, rots2, trans2, bp1, bp2, 
                    center=True, squeeze=False):
    import torch
    from sinc.info.joints import smpl2gpt, gpt_side, smpl_bps
    from sinc.info.joints import smpl_bps_ids_list, smpl_bps_ids, smplh_joints
    from sinc.info.joints import smpl_bps_ids

    # minimum bp should be 2
    if sum(bp1) < sum(bp2):
        bp1, bp2 = bp2, bp1
        rots1, rots2 = rots2, rots1
        trans1, trans2 = trans2, trans1

    if squeeze:
        rots1, trans1 = torch.squeeze(rots1), torch.squeeze(trans1)
        rots2, trans2 = torch.squeeze(rots2), torch.squeeze(trans2)

    # STEP 1: same length with centering
    # common length
    length_1 = len(rots1)
    length_2 = len(rots2)

    # assumption 1
    # should be the same lenght
    length = min(length_1, length_2)

    if center:
        # force lenght constraint to be centered
        start_1 = (length_1 - length)//2
        rots1 = rots1[start_1:start_1+length]
        trans1 = trans1[start_1:start_1+length]

        start_2 = (length_2 - length)//2
        rots2 = rots2[start_2:start_2+length]
        trans2 = trans2[start_2:start_2+length]
    else:
        # trim length
        rots1 = rots1[:length]
        trans1 = trans1[:length]
        rots2 = rots2[:length]
        trans2 = trans2[:length]

    # assumption 2:
    # For composition, the two legs + global should be packed together
    left_leg_id = smpl_bps_ids_list.index("left leg")
    right_leg_id = smpl_bps_ids_list.index("right leg")
    global_id = smpl_bps_ids_list.index("global")

    if bp2[left_leg_id] or bp2[right_leg_id] or bp2[global_id]:
        bp1[left_leg_id] = 0
        bp1[right_leg_id] = 0
        bp1[global_id] = 0
        bp2[left_leg_id] = 1
        bp2[right_leg_id] = 1
        bp2[global_id] = 1
    else:
        bp1[left_leg_id] = 1
        bp1[right_leg_id] = 1
        bp1[global_id] = 1
        bp2[left_leg_id] = 0
        bp2[right_leg_id] = 0
        bp2[global_id] = 0
    # bp2 is minimum, will be added at the end (override)
    # add more to bp1

    # assumption 3:
    # binary selection of everything
    for i, x2 in enumerate(bp2):
        if x2 == 0:
            bp1[i] = 1

    body_parts_1 = [smpl_bps_ids_list[i] for i, x in enumerate(bp1) if x == 1]
    body_parts_1 = [y for x in body_parts_1 for y in smpl_bps[x]]

    body_parts_2 = [smpl_bps_ids_list[i] for i, x in enumerate(bp2) if x == 1]
    body_parts_2 = [y for x in body_parts_2 for y in smpl_bps[x]]

    # STEP 2: extract the body_parts
    joints_1 = [smplh_joints.index(x) for x in body_parts_1]
    joints_2 = [smplh_joints.index(x) for x in body_parts_2]

    frank_rots = torch.zeros_like(rots1)
    frank_rots[:, joints_1] = rots1[:, joints_1]
    frank_rots[:, joints_2] = rots2[:, joints_2]

    # assumption 3
    # gravity base hand crafted translation rule
    if "foot" in " ".join(body_parts_1):
        frank_trans = trans1
    else:
        frank_trans = trans2

    return frank_rots, frank_trans

def transform_batch_to_mixed_synthetic(batch):
    rots_lst = []
    trans_lst = []
    lens_lst = []
    for idx, x in enumerate(batch['rots_a']):
        if 'synth' in batch['keyid'][idx]:
            rots_a = batch['rots_a'][idx]
            rots_b = batch['rots_b'][idx]
            trans_a = batch['trans_a'][idx]
            trans_b = batch['trans_b'][idx]
            rots_comb, trans_comb = gpt_compose(rots_a, trans_a,
                                                rots_b, trans_b, 
                                                batch['bp-gpt'][idx][0],
                                                batch['bp-gpt'][idx][1],
                                                center=False)
        else:
            rots_comb = batch['rots_a'][idx]
            trans_comb = batch['trans_a'][idx]
        curlen = len(rots_comb) 

        rots_lst.append(rots_comb)
        trans_lst.append(trans_comb)
        lens_lst.append(curlen)
    return rots_lst, trans_lst, lens_lst


from sinc.data.tools.collate import collate_tensor_with_padding
# lens, motions_ds = self.transform_batch_to_mixed_synthetic(batch)
dataset = BABEL(datapath='data/babel/babel-smplh-30fps-male',
                synthetic=True)
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0,
                        collate_fn=collate_rots_and_text)
for bs in dataloader:
    R_m, tr_m, lens = transform_batch_to_mixed_synthetic(bs)
    R_m = [rm[:ll] for rm, ll in zip(R_m, lens)]
    tr_m = [trm[:ll] for trm, ll in zip(tr_m, lens)]