
 

<p align="center">

  <h1 align="center">SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation
    <br>
    <a href='https://arxiv.org/abs/2304.10417'>
    <img src='https://img.shields.io/badge/arxiv-report-red' alt='ArXiv PDF'>
    </a>
    <a href='https://sinc.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </h1>
  <p align="center">
    <a href="https://ps.is.mpg.de/person/nathanasiou"><strong>Nikos Athanasiou*</strong></a>
    ·
    <a href="https://mathis.petrovich.fr"><strong>Mathis Petrovich*</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    ·
    <a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>
  </p>
  <h2 align="center">arXiv 2023</h2>
 <div align="center">Official PyTorch implementation of the paper "SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation" </div>
 <div align="center">
  </div>
</p>
<p float="center">
  <div align="center">
  <img src="assets/sinc_tsr.gif" />
  </div>
</p>

<!-- | Paper Video                                                                                                | Qualitative Results                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/vidid/0.jpg)](https://www.youtube.com/) | -->

## Features


This implementation:
- Instruction on how to prepare the datasets used in the experiments.
- The training code:
  - For SINC method
  - For the baselines 
  - For the ablations done in the paper

<h2 align="center">Environment & Basic Setup</h2>

<details>
  <summary>Details</summary>
SINC has been implemented and tested on Ubuntu 20.04 with python >= 3.10.

Clone the repo:
```bash
git clone https://github.com/athn-nik/sinc.git
```

After it do this to install DistillBERT:

```shell
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

Install the requirements using `virtualenv` :
```bash
# pip
source scripts/install.sh
```
You can do something equivalent with `conda` as well.
</details>



[comment]: <> (## Running the Demo)

[comment]: <> (We have prepared a nice demo code to run SINC on arbitrary videos. )



<h2 align="center">Data & Training</h2>

 <details>
  <summary>Details</summary>

<div align="center"><em>There is no need to do this step if you have followed the instructions and have done it for TEACH. Just use the ones from TEACH.</em></div>

<div align="center"><h3>Step 1: Data Setup</h3></center></div>

Download the data from [AMASS website](https://amass.is.tue.mpg.de). Then, run this command to extract the amass sequences that are annotated in babel:

```shell
python scripts/process_amass.py --input-path /path/to/data --output-path path/of/choice/default_is_/babel/babel-smplh-30fps-male --use-betas --gender male
```

Download the data from [TEACH website](https://teach.is.tue.mpg.de), after signing in. The data SINC was trained was a processed version of BABEL. Hence, we provide them directly to your via our website, where you will also find more relevant details. 
Finally, download the male SMPLH male body model from the [SMPLX website](https://smpl-x.is.tue.mpg.de/). Specifically the AMASS version of the SMPLH model. Then, follow the instructions [here](https://github.com/vchoutas/smplx/blob/main/tools/README.md#smpl-h-version-used-in-amass) to extract the smplh model in pickle format.

The run this script and change your paths accordingly inside it extract the different babel splits from amass:

```shell
python scripts/amass_splits_babel.py
```

Then create a directory named `data` and put the babel data and the processed amass data in.
You should end up with a data folder with the structure like this:

```
data
|-- amass
|  `-- your-processed-amass-data 
|
|-- babel
|   `-- babel-teach
|       `...
|   `-- babel-smplh-30fps-male 
|       `...
|
|-- smpl_models
|   `-- smplh
|       `--SMPLH_MALE.pkl
```

Be careful not to push any data! 
Then you should softlink inside this repo. To softlink your data, do:

`ln -s /path/to/data`

You can do the same for your experiments:

`ln -s /path/to/logs experiments`

Then you can use this directory for your experiments.

<div align="center"><h3>Step 2: Training</h3></center></div>

To start training after activating your environment. Do:

```shell
python train.py experiment=baseline logger=none
```

Explore `configs/train.yaml` to change some basic things like where you want
your output stored, which data you want to choose if you want to do a small
experiment on a subset of the data etc.
You can disable the text augmentations and using `single_text_desc: false` in the
model configuration file. You can check the `train.yaml` for the main configuration
and this file will point you to the rest of the configs (eg. `model` refers to a config found in
the folder `configs/model` etc.).

</details>

<h2 align="center"> Evaluation</h2>

<details>
  <summary>Details</summary>

After training, to sample and evaluate a model which has been stored in a folder `/path/to/experiment`
``` bash
python sample.py folder=/path/to/experiment/ ckpt_name=699 set=small

python eval.py folder=/path/to/experiment/ ckpt_name=699 set=small
```

- You can change the `jointstype` for the sampling script to output and save rotations and translation by setting `joinstype=rots`.
- By setting the `set=full` you will obtain the results on the full BABEL validation set.

You can calculate the TEMOS score using: 

``` bash
python sample_eval_latent.py folder=/is/cluster/fast/nathanasiou/logs/space/single-text-baselines/rs_only/babel-amass/ ckpt_name=699 set=small
```
</details>

 
## Citation

```bibtex
@inproceedings{SINC:ICCV:2022,
  title={{SINC}: Spatial Composition of {3D} Human Motions for Simultaneous Action Generation},
  author={Athanasiou, Nikos and Petrovich, Mathis and Black, Michael J. and Varol, G\"{u}l },
  booktitle = {arXiv},
  year = {2023}
}

```
## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.

## References
Many part of this code were based on the official implementation of [TEMOS](https://github.com/Mathux/TEMOS).

## Contact

This code repository was implemented by [Nikos Athanasiou](https://is.mpg.de/~nathanasiou) and [Mathis Petrovich](https://mathis.petrovich.fr/).

Give a ⭐ if you like.

For commercial licensing (and all related questions for business applications), please contact ps-licensing@tue.mpg.de.
