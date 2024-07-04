from pathlib import Path

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


model_dir = Path("/srv/beegfs/scratch/groups/rodem/jetssl/jetssl2/")
target_dir = Path("/srv/beegfs/scratch/groups/rodem/jetssl/jetssl3/backbones/")

model_names = ["reg4", "diff4", "flow4", "vae4", "kmeans4", "mdm4", "gpt4"]
