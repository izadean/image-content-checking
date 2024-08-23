import json
import os.path

import click
import pandas as pd


def parse_images(embeddings_file: str, images: dict, annotations: dict) -> pd.DataFrame:
    with open(embeddings_file, "r") as f:
        embeddings = json.load(f)
    records = []
    images = {image["id"]: image for image in images}
    annotations = {str(ann["id"]): ann for ann in annotations}
    for embeddings_id in embeddings.keys():
        annotation = annotations[embeddings_id]
        image = images[annotation["image_id"]]
        records.append([image["file_name"], embeddings_id])
    return pd.DataFrame(records, columns=["file_name", "embeddings_id"])


def read_coco_images(annotations_file: str) -> tuple[dict, dict]:
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    return coco_data["images"], coco_data["annotations"]


def shuffle_df(df_final):
    df_final = df_final.sample(frac=1, random_state=10).reset_index(drop=True)
    return df_final


def add_false_class(df):
    df["label"] = 1
    df_duplicate = df.copy()
    df_duplicate["file_name"] = shuffle_df(df_duplicate["file_name"])
    df_duplicate["label"] = 0
    df_combined = pd.concat([df, df_duplicate]).reset_index(drop=True)
    return shuffle_df(df_combined)


@click.command
@click.argument("annotations-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("embeddings-file", type=click.Path(exists=True, dir_okay=False))
@click.argument("target-file", type=click.Path(exists=False, dir_okay=False))
def main(annotations_dir: str, embeddings_file: str, target_file: str) -> None:
    assert target_file[-3:] == "csv", "target-file should be of type CSV"
    df1 = parse_images(embeddings_file, *read_coco_images(os.path.join(annotations_dir, "captions_val2014.json")))
    df2 = parse_images(embeddings_file, *read_coco_images(os.path.join(annotations_dir, "captions_train2014.json")))
    df_final = add_false_class(pd.concat([df1, df2]))
    df_final.to_csv(target_file)


if __name__ == "__main__":
    main()
