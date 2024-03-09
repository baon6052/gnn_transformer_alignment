from os import listdir
from os.path import isfile, join

import click
import jax

from checkpointer import Checkpointer


@click.command()
@click.option("--model_save_path", type=str, default="trained_models")
def get_total_number_of_parameters(model_save_path: str = "trained_models"):

    for filename in listdir(model_save_path):
        file_path = join(model_save_path, filename)
        if isfile(file_path):
            checkpointer = Checkpointer(file_path)
            parameters = checkpointer.load()
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(parameters))
            print(f"{filename=} {param_count=}")

if __name__ == "__main__":
    get_total_number_of_parameters()
