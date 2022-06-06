import os

import neptune.new as neptune

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

USE_CUDA = int(os.getenv("USE_CUDA", 0))

# FIXME: This is hacky and must be changed. Neptune doesn't allow to upload files to another directory unless we assume
#        a different working directory
original_cwd = os.getcwd()
# FIXME: It should be an ENV variable
os.chdir('/data' if USE_CUDA else '.')
NEPTUNE_INSTANCE = neptune.init(project="jbsimoes/graph-rl",
                                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYmQ4MjE1OC0yNzBhLTQyNzctYjFmZS00YTFhYjYxZTdmMjUifQ==",
                                source_files=['*.py'],
                                mode=os.getenv("NEPTUNE_MODE", "async"))
os.chdir(original_cwd)
