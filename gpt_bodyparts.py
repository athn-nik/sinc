import os
import openai
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import sinc.launch.prepare  # noqa
from hydra.utils import to_absolute_path
from pathlib import Path
from tqdm import tqdm
from sinc.utils.file_io import read_json, write_json
import time
from rich import print 

logger = logging.getLogger(__name__)
openai.api_key = '<api-key>'

@hydra.main(config_path="configs", config_name="gpt_bdpts")
def _gpt_extract(cfg: DictConfig):
    return gpt_extract(cfg)

# body_parts = ['left arm', 'right arm', 'left leg', 'global orientation',
# 'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
# 'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

# # fine_bp = list(body_parts)
# coarse_bp = ['left arm', 'right arm', 'left leg', 'global orientation', 'right leg', 'torso']
# coarse_bp_v2 = ['left arm', 'right arm', 'left leg', 'global orientation',
#                 'right leg', 'torso', 'head', 'neck', 'pelvis']

# 'What parts of the body are moving when someone is doing the action:'


def gpt_extract(cfg: DictConfig):
 
    from sinc.utils.file_io import write_json
    from sinc.utils.text_constants import prompts_to_gpt, final_prompts_to_gpt
    from sinc.utils.text_constants import unique_texts_babel_train_val
    unique_texts = list(unique_texts_babel_train_val)

    responses = {}
    responses_full = {}
    write_every = 20
    elems = 0
    unique_texts = unique_texts[6360:]
    models_GPT = {'curie':'text-curie-001',
                  'ada': 'text-ada-001',
                  'davinci': 'text-davinci-003'
                  }
    assert cfg.gpt_model in models_GPT.keys()
    
    gpt_model = models_GPT[cfg.gpt_model]
    n_to_keep = cfg.ntokeep if cfg.ntokeep is not None else len(unique_texts)
 
    if cfg.crawl_data:
        for action_text in tqdm(unique_texts[:n_to_keep]):
            prompts = [p.replace('[ACTION]', action_text) for p in final_prompts_to_gpt]
            response_compl = openai.Completion.create(model=gpt_model,
                                                      prompt=prompts,
                                                      temperature=0.0,
                                                      max_tokens=256,
                                                      top_p=1,
                                                      frequency_penalty=0,
                                                      presence_penalty=0
                                                    )

            assert len(response_compl['choices']) == len(prompts)
            lofrespons = [response_compl['choices'][i]['text'].strip() for i in range(len(prompts))]
            responses[action_text] = lofrespons
            responses_full[action_text] = lofrespons
            elems += 1
            if elems % write_every == 0:
                batch_id = elems // write_every
                write_json(responses, f'gpt-labels_batch{batch_id}.json')
                responses = {}
                time.sleep(30)
            if elems == 100:
                time.sleep(120)
        # final batch
        batch_id += 1
        write_json(responses, f'gpt-labels_batch{batch_id}.json')
        # write html/tex to file
        pathout='gpt-labels_full.json'
        write_json(responses_full, pathout)

        # p2 = '/home/nathanasiou/Desktop/gpt3-labels.json'
        # write_json(responses_full, p2)
    else:
        # process
        html_toks = ['<tr>', '<td>', '<tr>', '</tr>', '</td>', '</tr>', '<th>', '</th>']

        p = '/home/nathanasiou/Desktop/example-batch.html'
        all_texts = []
        with open(p, 'r') as f:
            for l in f:
                for tk in html_toks:
                    l = l.strip().replace(tk, '')
                text = l
                if text:
                    all_texts.append(text)
        actions = all_texts[::3]
        p1 = all_texts[1::3]
        p2 = all_texts[2::3]
        import spacy
        nlp = spacy.load("en_core_web_sm")
        pipeline = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
        from space.info.joints import bp_gpt
        p1_annot = {k:[] for k in p1}
        p2_annot = {k:[] for k in p2}
        for alt, annot in zip([p1, p2], [p1_annot, p2_annot]):
            for doc in nlp.pipe(alt, disable=["tok2vec", "ner"]):
                for i in doc:
                    POS = i.pos_
                    if i.text.lower() in bp_gpt:                
                        annot[doc.text].append(i.text)
        from space.utils.file_io import write_json
        act_labels = {k: [] for k in actions}
        p1_bps = list(p1_annot.values())
        p2_bps = list(p2_annot.values())

        for i, act in enumerate(actions):
            act_labels[act].append(list(set(p1_bps[i])))
            act_labels[act].append(list(set(p2_bps[i])))

        write_json(act_labels, './acts.json')
        write_json(p1_annot, './promp1.json')
        write_json(p2_annot, './promp2.json')

             
if __name__ == '__main__':
    _gpt_extract()
