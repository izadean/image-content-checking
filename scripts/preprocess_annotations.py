import os.path
from functools import lru_cache
import json

import click
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

TARGET_WIDTHS_AND_HEIGHTS_PAIRS = [
    (640, 480),
    (480, 640),
    (427, 640),
    (640, 426),
    (640, 427),
    (640, 428),
]


def read_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def write_json(result: dict, target_file):
    with open(target_file, "w") as f:
        json.dump(result, f, indent=4)


def is_valid_size(image: dict) -> bool:
    return (image["width"], image["height"]) in TARGET_WIDTHS_AND_HEIGHTS_PAIRS


@lru_cache
def get_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer("avsolatorio/GIST-small-Embedding-v0", revision=None)


def embed(text: list[str]) -> np.ndarray:
    return get_sentence_transformer().encode(text, convert_to_tensor=False)


def parse_annotations(annotations: dict, images: dict):
    images = {image['id']: image for image in images}
    captions = []
    annotations_ids = []
    for annotation in annotations:
        image = images[annotation["image_id"]]
        if not is_valid_size(image):
            continue
        captions.append(annotation["caption"])
        annotations_ids.append(annotation["id"])
    
    result = dict()
    for annotations_id, embeddings in zip(annotations_ids, embed(captions)):
        result[annotations_id] = embeddings.tolist()
    return result


def read_coco_annotations(annotations_file: str):
    coco_data = read_json(annotations_file)
    return coco_data["annotations"], coco_data["images"]


def merge_annotations(annotations1: dict, annotations2: dict):
    for k, v in annotations2.items():
        annotations1[k] = v
    return annotations1


@click.command
@click.argument("annotations-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("target-file", type=click.Path(exists=False, dir_okay=False))
def main(annotations_dir: str, target_file: str) -> None:
    assert target_file[-4:] == "json", "target-file must be of type JSON"
    anns1 = parse_annotations(*read_coco_annotations(os.path.join(annotations_dir, "captions_train2014.json")))
    anns2 = parse_annotations(*read_coco_annotations(os.path.join(annotations_dir, "captions_val2014.json")))
    result = merge_annotations(anns1, anns2)
    write_json(result, target_file)


if __name__ == '__main__':
    main()
