docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
-v /home/fenix/dev/evoarc:/evoarc \
-v /home/fenix/dev/evoarc/data:/kaggle/input/arc-prize-2024 \
-v /home/fenix/dev/evoarc/output:/kaggle/working \
-e COMPUTE_BACKEND="fenix" \
-e MORPH=$1 \
-e WANDB_API_KEY=$WANDB_API_KEY \
nvcr.io/nvidia/jax:24.10-py3 \
bash -c "./evoarc/scripts/\$COMPUTE_BACKEND/setup.sh && jupyter nbconvert --to notebook --execute /evoarc/morphs/\$MORPH.ipynb --stdout"