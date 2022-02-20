import os

import neptune.new as neptune

NEPTUNE_INSTANCE = neptune.init(project="jbsimoes/graph-rl",
                                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYmQ4MjE1OC0yNzBhLTQyNzctYjFmZS00YTFhYjYxZTdmMjUifQ==",
                                source_files=['*.py'],
                                mode=os.getenv("NEPTUNE_MODE", "async")
                                )

USE_CUDA = os.getenv("USE_CUDA", 0)
