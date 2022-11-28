import os

USE_CUDA = int(os.getenv("USE_CUDA", 0))

original_cwd = os.getcwd()

# FIXME: Move it to ENV variable. I can't do it now because I don't want to deal with the DOCKER rebuild process.
WANDB_PATH = '/data' if USE_CUDA == 1 else '.'

"""
Uncomment this line if you want to use Neptune.

os.chdir(WANDB_PATH)

NEPTUNE_INSTANCE = neptune.init(project="jbsimoes/graph-rl",
                                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYmQ4MjE1OC0yNzBhLTQyNzctYjFmZS00YTFhYjYxZTdmMjUifQ==",
                                source_files=['*.py'],
                                mode=os.getenv("NEPTUNE_MODE", "async"),
                                name=os.getenv("NEPTUNE_RUN_NAME", None))

os.chdir(original_cwd)
"""

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
