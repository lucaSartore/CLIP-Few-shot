from math import e
from typing import Any, Iterable
from torch.nn.functional import triplet_margin_with_distance_loss
from torch.utils.data import Subset
from torch import Tensor
from torchvision.datasets import Flowers102
import torch
from classes_names import CLASS_NAMES
import clip

def base_novel_categories(dataset: Flowers102):
    # set returns the unique set of all dataset classes
    all_classes = set(dataset._labels)
    # and let's count them
    num_classes = len(all_classes)

    # here list(range(num_classes)) returns a list from 0 to num_classes - 1
    # then we slice the list in half and generate base and novel category lists
    base_classes = list(range(num_classes))[:num_classes//2]
    novel_classes = list(range(num_classes))[num_classes//2:]
    return base_classes, novel_classes


def get_data( clip_visual_model: str, device: str, data_dir="./dataset/flowers",) -> tuple[Flowers102, Flowers102, Flowers102]:

    _, preprocess = clip.load(clip_visual_model, device=device)

    train = Flowers102(root=data_dir, split="train", download=True, transform=preprocess )
    val = Flowers102(root=data_dir, split="val", download=True, transform=preprocess)
    test = Flowers102(root=data_dir, split="test", download=True, transform=preprocess)
    return train, val, test




def split_data(dataset: Flowers102, base_classes: Iterable[Any]) -> tuple[Subset[tuple[Tensor, int]], Subset[tuple[Tensor, int]]]:
    # these two lists will store the sample indexes
    base_categories_samples = []
    novel_categories_samples = []

    # we create a set of base classes to compute the test below in O(1)
    # this is optional and can be removed
    base_set = set(base_classes)

    # here we iterate over sample labels and also get the correspondent sample index
    for sample_id, label in enumerate(dataset._labels):
        if label in base_set:
            base_categories_samples.append(sample_id)
        else:
            novel_categories_samples.append(sample_id)

    # here we create the dataset subsets
    # the torch Subset is just a wrapper around the dataset
    # it simply stores the subset indexes and the original dataset (your_subset.dataset)
    # when asking for sample i in the subset, torch will look for its original position in the dataset and retrieve it
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
    base_dataset = torch.utils.data.Subset(dataset, base_categories_samples)
    novel_dataset = torch.utils.data.Subset(dataset, novel_categories_samples)
    return base_dataset, novel_dataset


def get_data_splitted(clip_mode: str, device: str):
    train_set, val_set, test_set = get_data(clip_mode, device)
    base_classes, novel_classes = base_novel_categories(train_set)

    train_base, _ = split_data(train_set, base_classes)
    val_base, _ = split_data(val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)

    return train_base, val_base, test_base, test_novel

if __name__ == "__main__":
    get_data_splitted("RN50", "cuda")
