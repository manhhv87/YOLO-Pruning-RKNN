# split_c2a_dataset.py
"""
Chia dataset C2A (images/ + labels/) thành train/val/test theo tỉ lệ 60:20:20.

Trước:
    dataset_root/
        images/*.jpg
        labels/*.txt

Sau:
    dataset_root/
        images/
            train/*.jpg
            val/*.jpg
            test/*.jpg
        labels/
            train/*.txt
            val/*.txt
            test/*.txt

Sử dụng:
    python split_c2a_dataset.py --dataset-root /path/to/C2A_synth --seed 42 --move
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs: List[Tuple[Path, Path]] = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"[WARN] Không tìm thấy label cho ảnh: {img_path.name}, bỏ qua.")
            continue
        pairs.append((img_path, lbl_path))

    return pairs


def split_indices(n: int, seed: int = 42):
    """
    Chia n phần tử thành train/val/test với tỉ lệ 60:20:20 (xấp xỉ).
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx


def copy_or_move(src: Path, dst: Path, move: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser(
        description="Chia C2A dataset thành train/val/test (60:20:20)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Thư mục gốc dataset (chứa images/ và labels/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed cho random shuffle (default: 42).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Di chuyển file thay vì copy (mặc định: copy).",
    )

    args = parser.parse_args()

    root = Path(args.dataset_root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError("Không tìm thấy images/ hoặc labels/ trong dataset_root.")

    pairs = collect_pairs(images_dir, labels_dir)
    n = len(pairs)
    if n == 0:
        raise RuntimeError("Không tìm thấy cặp (image, label) nào.")

    print(f"Tổng số mẫu: {n}")

    train_idx, val_idx, test_idx = split_indices(n, seed=args.seed)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, idx_list in splits.items():
        for idx in idx_list:
            img_path, lbl_path = pairs[idx]

            dst_img = root / "images" / split_name / img_path.name
            dst_lbl = root / "labels" / split_name / lbl_path.name

            copy_or_move(img_path, dst_img, move=args.move)
            copy_or_move(lbl_path, dst_lbl, move=args.move)

    print("Hoàn tất chia train/val/test.")


if __name__ == "__main__":
    main()
