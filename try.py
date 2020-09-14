import json
from config.config import cfg

cfg.merge_from_file('./config/config.yaml')
cfg.freeze()
print(json.dumps(cfg, indent=4))

