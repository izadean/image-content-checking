import os
import json
from typing import Callable
from PIL import Image
import pandas as pd
import click


def load_annotations(annotations_dir: str) -> pd.DataFrame:
    anns1 = load_annotations_file(os.path.join(annotations_dir, "captions_val2014.json"))
    anns2 = load_annotations_file(os.path.join(annotations_dir, "captions_train2014.json"))
    return pd.concat([anns1, anns2])


def load_annotations_file(annotations_file: str) -> pd.DataFrame:
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    data = [
        [image['file_name'], image['width'], image['height']]
        for image in (coco_data['images'])
    ]
    return pd.DataFrame(data, columns=['file_name', 'width', 'height'])


def transform_image(src: str, dst: str, transform_fn: Callable[[Image.Image], Image.Image]) -> None:
    with Image.open(src) as img:
        transform_fn(img).save(dst)


def rotate_image_right(src: str, dst: str) -> None:
    transform_image(src, dst, lambda image: image.rotate(-90, expand=True))


def resize_image_640x480(src: str, dst: str) -> None:
    transform_image(src, dst, lambda image: image.resize((640, 480)))


def rotate_right_and_resize_image_640x480(src: str, dst: str) -> None:
    transform_image(src, dst, lambda image: image.rotate(-90, expand=True).resize((640, 480)))


def get_filtered_images_in(df: pd.DataFrame, target_widths: list[int], target_heights: list[int]) -> pd.Series:
    df_filter = lambda rec: rec["width"] in target_widths and rec["height"] in target_heights
    projection = df.apply(df_filter, axis=1)
    return df[projection]


@click.command
@click.argument("annotations-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("images-src", type=click.Path(exists=True, file_okay=False))
@click.argument("images-dst", type=click.Path(exists=True, file_okay=False))
def main(annotations_dir: str, images_src: str, images_dst: str) -> None:
    annotations = load_annotations(annotations_dir)
    annotations["src"] = annotations["file_name"].apply(lambda f: os.path.join(images_src, f))
    annotations["dst"] = annotations["file_name"].apply(lambda f: os.path.join(images_dst, f))

    get_filtered_images_in(annotations, [640], [426, 427, 428]).apply(
        lambda rec: resize_image_640x480(rec["src"], rec["dst"]),
        axis=1
    )

    get_filtered_images_in(annotations, [480], [640]).apply(
        lambda rec: rotate_image_right(rec["src"], rec["dst"]),
        axis=1
    )

    get_filtered_images_in(annotations, [427], [640]).apply(
        lambda rec: rotate_right_and_resize_image_640x480(rec["src"], rec["dst"]),
        axis=1
    )


if __name__ == "__main__":
    main()
