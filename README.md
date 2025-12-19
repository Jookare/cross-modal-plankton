# Cross-modal learning for plankton recognition

Code and models for the paper: "Cross-modal learning for plankton recognition"

The project focuses on joint image-signal representation learning for plankton recognition using CLIP-style contrastive pre-training. 

The code was built using 
```
Python 3.12.3
PyTorch 2.5.1
CUDA 13.0
```

# Usage
To easily use these codes install the project in editable mode, by running from project root
```
pip install -e .
```
This puts the project to `PYTHONPATH` and fixes imports from `src`.

## Downloading the datasets

The datasets can be downloaded from here `https://ida.fairdata.fi/s/NOT-FOR-PUBLICATION-HcirdqiwoMzb`. It includes LAB, SEA, and UTO datasets that were used in the publication. Move dataset folders to the `data` folder.

## Training models
Model architectures and training settings are defined using model cards in `configs/train/[multi/dino]`

Model card specifies:
- Image encoder (e.g. ConvNeXt, ViT, EfficientNet)
- Profile encoder
- Projection heads
- Training hyperparameters
- Loss configuration

### Cross-modal training
Image and profile encoders can be jointly trained using:
```
python scripts/train_multi.py --config configs/train/multi/your_model.yaml -d data/<DATASET>
```
or if training multiple models via shell scripts
```
bash scripts/train_multi.sh
```
The checkpoints and logs are written to `logs/<experiment_name>` and can be analyzed in tensorboard by running `tensorboard --logdir=logs`.

### Image-only training
To train an image-only self-supervised baseline run:
```
python scripts/train_dino.py --config configs/train/dino/your_model.yaml -d data/<DATASET>
```
or
```
bash scripts/train_dino.sh
```

The DINO implementation uses `LightlySSL` python package of which more information can be found from here [Lightly Github](https://github.com/lightly-ai/lightly)


## Extracting embeddings
After training, embeddings need to be extracted for evaluation.

This also follows similar setup from training, so first define an experiment `configs/experiment/experiment.yaml`.

This file specifies:
- Which trained models to use
- Which dataset to run on
- Image size, batch size, number of workers for dataloader
- Where to save embeddings

Afterwards extract embeddings by running:
```
python scripts/extract_embeddings.py
```

This saves the embeddings to `results/embeddings/` as `.pkl`.

### Benchmarking (kNN evaluation)
To evaluate embeddings using kNN:
```
python scripts/run_benchmark.py
```
The benchmark:

- Samples gallery sets of varying size per class
- Evaluates image-only, profile-only, and combined representations
- Supports multiple values of k
- Selects gallery set multiple times for robustness

This saves the results to `results/benchmarks/` as `.pkl`.

### Analyzing results
The results can be analyzed in the provided jupyter notebook `analyze_results.ipynb`.

![](results/figures/example_results2.png)
