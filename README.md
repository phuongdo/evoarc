# evoarc 🧫🔬

**Evo**lutionary Code generation for **ARC**-AGI Kaggle Challenge

## Abstract

uses an agent (e.g. claude sonnet, gpt 4o, etc) to rewrite the model and training pipeline for a deep-learning notebook solution to the arc-agi 2024 kaggle challenge. solutions, called `morphs` are run in docker containers and compete to get the highest eval accuracy. a basic evolutionary algorithm with mutation and selection improves the morphs over time.

various types of compute were used

- `ojo` - a local nvidia agx orin
- `oop` - a local linux pc with nvidia 3090
- `big` - a cloud instance with 1xH100
- `kag` - a cloud kaggle nb with 4xT4

## Setup

first ssh into your machine and setup the environment (must have docker installed)

```bash
git clone https://github.com/hu-po/evoarc
pip install nbformat arxiv wandb
export WANDB_API_KEY=...
export WANDB_MODE=disabled
```

depending on what models you want to use as the agent, setup your keys:

```bash
pip install replicate
export REPLICATE_API_TOKEN=...
pip install openai 
export OPENAI_API_KEY=...
pip install anthropic
export ANTHROPIC_API_KEY=...
```

## Usage

```bash
nvidia-smi --list-gpus
GPU 0: NVIDIA H100 (UUID: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
GPU 1: NVIDIA A100 (UUID: GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy)

docker run --rm --gpus '"device=7"' nvcr.io/nvidia/jax:24.04-py3 nvidia-smi
```

run the morph `foo` locally on `fenix`:

```bash

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export WANDB_MODE=offline
./scripts/test.sh fenix cnn.o1
```

start the evolutionary process using `foo` as the protomorph:

```bash
python3 evolve.py --seed 42 --compute_backend fenix --protomorphs foo
```

mutate a single morph:

```bash
python3 mutate.py --morph foo
```

clean out the output directory:

```bash
./scripts/clean.sh
```

to submit a morph to kaggle you need to "create a notebook" from the ["code" page](https://www.kaggle.com/competitions/arc-prize-2024/code) and paste in the code for your morph from `output/foo/export.ipynb`. then click "save version" and make sure to disable internet. then go to the ["submit" page](https://www.kaggle.com/competitions/arc-prize-2024/submit) and hit "submit prediction".


## Dockerbuild

```bash
docker build -t evoarc .
```


## References

ARC-AGI Challenge
- https://arcprize.org/
- https://www.kaggle.com/competitions/arc-prize-2024
- https://x.com/fchollet
- https://x.com/arcprize

CAX: Cellular Automata Accelerated in JAX
- https://arxiv.org/pdf/2410.02651.pdf
- https://github.com/maxencefaldor/cax

Tackling the Abstraction and Reasoning Corpus with Vision Transformers: the Importance of 2D Representation, Positions, and Objects
- https://arxiv.org/abs/2410.06405

Lambda Cloud
- https://cloud.lambdalabs.com/instances

Dev
- https://uithub.com/
- https://chatgpt.com/
- https://claude.ai/
- https://wandb.ai/hug/evoarc

Videos
-https://youtu.be/3ZTNps2PraM

## Video

[![YouTube Video](https://img.youtube.com/vi/9J1Ofd1gYIk/0.jpg)](https://www.youtube.com/watch?v=9J1Ofd1gYIk)

## Citation

```
@misc{hupo2024evoarc,
  title={EvoARC: Evolutionary Code generation for ARC-AGI Kaggle Challenge},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/hu-po/evoarc}
}
```
