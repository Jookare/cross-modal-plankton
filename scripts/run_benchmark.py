import pickle
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
from src.benchmark import benchmark
from pathlib import Path

# Threshold filtering
def threshold(data, coder, th):
    images, profiles, names = data
    label = coder.transform(names)
    uniqs, counts = np.unique(label, return_counts=True)
    mask = counts >= th
    hits = tuple()
    for id in uniqs[mask]:
        hits += np.where(label == id)
    hits = np.concatenate(hits)
    return images[hits], profiles[hits], names[hits]

def main():
    cfg = yaml.safe_load(open("configs/experiment/experiment.yaml"))

    # Minimum samples per class
    min_samples = cfg["benchmark"]["min_samples"]
    seed = cfg["benchmark"]["seed"]
    repeats = cfg["benchmark"]["repeats"]
    k_values = cfg["benchmark"]["k_values"]
    n_per_class = cfg["benchmark"]["n_per_class"]
    
    with open(cfg["embedding"]["save_path"], "rb") as f:
        embeddings = pickle.load(f)
    
    # Initialize label encoder
    key_ = list(embeddings.keys())[0]
    coder = LabelEncoder().fit(embeddings[key_][1]['classes'])
    print(f"First key: {key_}")
    results = {name: {} for name in embeddings.keys()}  

    for name, data in tqdm(embeddings.items()):
        print("running results for model", name)
        for fold in data.keys():
            results[name][fold] = {}
            foo = data[fold]
        
            # Check if data has train/test splits
            has_split = "train" in foo and "test" in foo
            if has_split:
                train = (
                    foo["train"]["image"],
                    foo["train"]["profile"],
                    foo["train"]["label"],
                )
                test = (
                    foo["test"]["image"],
                    foo["test"]["profile"],
                    foo["test"]["label"],
                )
                bar = (train, test)
            else:
                bar = (foo['image'], foo['profile'], foo['label'])
                bar = threshold(bar, coder, min_samples)
                
            for n in n_per_class:
                # Set random seed for reproducibility
                random.seed(seed)
                np.random.seed(seed)
                subresults = benchmark(bar, coder, n, repeats, k_values)
                results[name][fold][n] = subresults

    save_path = Path(cfg["benchmark"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"saving results to {save_path}")
    with open(cfg["benchmark"]["save_path"], "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
