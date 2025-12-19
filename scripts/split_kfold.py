import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_kfold_split(annot, train_idx, temp_idx, seed):
    train_df = annot.iloc[train_idx]
    temp_df = annot.iloc[temp_idx]

    # Temp → 15% test, 5% val
    val_df, test_df = train_test_split(temp_df, test_size=0.75, stratify=temp_df["class"], random_state=seed)

    return (
        train_df.sort_index(),
        val_df.sort_index(),
        test_df.sort_index(),
    )


def make_single_split(annot, seed):
    train_idx, val_idx = train_test_split(annot.index, test_size=0.05, stratify=annot["class"], random_state=seed)

    return (
        annot.iloc[train_idx].sort_index(),
        annot.iloc[val_idx].sort_index(),
    )


def save_kfold_split(train_df, val_df, test_df, out_dir, stepback):
    out_dir.mkdir(exist_ok=True)

    for df in (train_df, val_df, test_df):
        df.loc[:, ["image", "profile"]] = df[["image", "profile"]].apply(lambda x: "../" * stepback + x)

    train_df.to_csv(out_dir / "train.csv")
    val_df.to_csv(out_dir / "val.csv")
    test_df.to_csv(out_dir / "test.csv")


def save_single_split(train_df, val_df, out_dir, stepback):
    out_dir.mkdir(exist_ok=True)

    for df in (train_df, val_df):
        df.loc[:, ["image", "profile"]] = df[["image", "profile"]].apply(lambda x: "../" * stepback + x)

    train_df.to_csv(out_dir / "train.csv")
    val_df.to_csv(out_dir / "val.csv")


def save_split(train_df, val_df, test_df, out_dir, stepback):
    out_dir.mkdir(exist_ok=True)

    for df in (train_df, val_df, test_df):
        df.loc[:, ["image", "profile"]] = df[["image", "profile"]].apply(lambda x: "../" * stepback + x)

    train_df.to_csv(out_dir / "train.csv")
    val_df.to_csv(out_dir / "val.csv")
    test_df.to_csv(out_dir / "test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["kfold", "single"],
        default="kfold",
        help="kfold or single train/val split",
    )
    parser.add_argument("-d", "--dataset", help="Path to the annotation file and directories images/ and profiles/")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-k", "--kfolds", type=int, default=5)
    parser.add_argument("-n", "--name", default="fold")
    args = parser.parse_args()

    data_dir = Path(args.dataset)
    annot = pd.read_csv(data_dir / "annotations.csv")

    # Adjusts paths to account for the folder structure
    stepback = args.name.count("/") + 1

    # kfold splitting
    if args.mode == "kfold":
        kfold = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

        for k, (train_idx, temp_idx) in enumerate(kfold.split(annot, annot["class"]), start=1):
            fold_dir = data_dir / f"{args.name}{k}"
            train_df, val_df, test_df = make_kfold_split(annot, train_idx, temp_idx, args.seed)

            save_kfold_split(train_df, val_df, test_df, fold_dir, stepback)

        print("Dataset folds created to annotation\n" + f"files {args.name}[1-{args.kfolds}]/[train/val/test].csv.")

    # single train / val split
    elif args.mode == "single":
        train_df, val_df = make_single_split(annot, args.seed)

        out_dir = data_dir / args.name
        save_single_split(train_df, val_df, out_dir, stepback)
        print("Dataset folds created to annotation\n" + f"files {args.name}/[train/val].csv.")

    else:
        raise ValueError(f"Undefined mode {args.mode}")
