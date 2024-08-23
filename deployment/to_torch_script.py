import click
import mlflow
import torch

from src.model import ContentCheckingModel


@click.command()
@click.argument("run-id", type=str)
@click.argument("model-name", type=str)
def main(run_id: str, model_name: str) -> None:
    model = mlflow.pytorch.load_checkpoint(ContentCheckingModel, run_id)
    model_input = (torch.rand((1, 3, 224, 224)), torch.rand((1, 384)))
    traced = torch.jit.trace(model, model_input)
    traced.save(f"deployment/scripted-models/{model_name}.pt")


if __name__ == '__main__':
    main()
