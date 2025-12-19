import numpy as np
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Sampling helper
def sample(y, n):
    idx = []
    orig = np.arange(len(y))
    for label in np.unique(y):
        i = list(orig[y == label])
        # Select all if total samples is less than gallery set
        if len(i) < n:
            j = i
        else:
            j = random.sample(i, n)
        idx.extend(j)
    return np.array(idx)


# Weighted mode computation
def weighted_mode(classes, weights, axis):
    """Compute weighted mode along axis"""
    unique_classes = np.unique(classes)
    votes = np.zeros((classes.shape[0], len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        mask = classes == cls
        votes[:, i] = np.sum(weights * mask, axis=axis)

    predictions = unique_classes[np.argmax(votes, axis=1)]
    return predictions, votes.max(axis=1)


# Predict k closest
def predict_k(predictor, coder, labels, keys, X_list, k_override=None):
    out = {}

    original_k = predictor.n_neighbors

    if k_override is not None:
        predictor.set_params(n_neighbors=k_override)

    for key, X in zip(keys, X_list):
        all_idx = []
        all_dist = []

        # Get neighbors from each input modality
        for X_test in X:
            dist, idx = predictor.kneighbors(X_test)
            all_idx.append(idx)
            all_dist.append(dist)

        # Combine neighbors from all modalities
        combined_idx = np.hstack(all_idx)
        combined_dist = np.hstack(all_dist)

        # Calculate weights (inverse distance)
        with np.errstate(divide="ignore"):
            weights = 1.0 / combined_dist
            inf_mask = np.isinf(weights)
            inf_row = np.any(inf_mask, axis=1)
            weights[inf_row] = inf_mask[inf_row].astype(float)

        neighbor_labels = labels[combined_idx]
        predictions, confidence = weighted_mode(neighbor_labels, weights, axis=1)

        out[key] = coder.inverse_transform(predictions)

    # Restore original neighbor count
    if k_override is not None:
        predictor.set_params(n_neighbors=original_k)

    return out


def benchmark(data, coder, n_per_class, repeats, K):
    has_split = isinstance(data[0], tuple) and isinstance(data[1], tuple)
    results = {}

    if has_split:
        # Predefined train/test
        train, test = data
        image_train_full, profile_train_full, name_train = train
        image_test, profile_test, name_test = test

        label_train_full = coder.transform(name_train)
        label_test = coder.transform(name_test)

    else:
        # No split
        images, profiles, names = data
        labels = coder.transform(names)

        length = len(labels)
        idx_full = set(range(length))

    for run in tqdm(range(repeats), desc=f"Repeats (gallery={n_per_class})", leave=False):
        if has_split:
            # Sample within the training split
            idx_train = sample(label_train_full, n_per_class)
            image_train = image_train_full[idx_train]
            profile_train = profile_train_full[idx_train]
            label_train = label_train_full[idx_train]

        else:
            # Random train/test split for this repeat
            idx_train = sample(labels, n_per_class)
            idx_test = list(idx_full - set(idx_train))

            image_train = images[idx_train]
            profile_train = profiles[idx_train]
            label_train = labels[idx_train]

            image_test = images[idx_test]
            profile_test = profiles[idx_test]
            label_test = labels[idx_test]

        # Evaluation
        results[run] = {
            "pred": {k: {} for k in K},
            "true": coder.inverse_transform(label_test),
        }

        # Init the NN predictor, n_neighbors is placeholder
        predictor = NearestNeighbors(n_neighbors=1, metric="cosine")
        predictor.fit(image_train)

        # Image only
        for k in K:
            results[run]["pred"][k] |= predict_k(
                predictor,
                coder,
                label_train,
                ("I - I", "I - P", "I - I+P"),
                ((image_test,), (profile_test,), (image_test, profile_test)),
                k_override=k,
            )

        # Profile only
        predictor.fit(profile_train)
        for k in K:
            results[run]["pred"][k] |= predict_k(
                predictor,
                coder,
                label_train,
                ("P - I", "P - P", "P - I+P"),
                ((image_test,), (profile_test,), (image_test, profile_test)),
                k_override=k,
            )

        # Image + Profile
        data_train_double = np.concatenate((image_train, profile_train))
        label_train_double = np.tile(label_train, (2,))
        predictor.fit(data_train_double, label_train_double)
        for k in K:
            results[run]["pred"][k] |= predict_k(
                predictor,
                coder,
                label_train_double,
                ("I+P - I", "I+P - P", "I+P - I+P"),
                ((image_test,), (profile_test,), (image_test, profile_test)),
                k_override=k,
            )

    return results
