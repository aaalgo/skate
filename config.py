DEFAULT_SEED = 2023
DEFAULT_DATALOADER_SEED = 2023
DEFAULT_SIZE = 400

try:
    from dist_config import *
except:
    pass

try:
    from local_config import *
    print("Found local config, importing...")
except:
    pass

