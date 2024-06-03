import os


def execute() -> None:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
