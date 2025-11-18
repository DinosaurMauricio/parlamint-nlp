import os
from wandb.sdk.lib.runid import generate_id

TXT_EXT = ".txt"
META_EXT = "-meta.tsv"
ANA_META_EXT = "-ana-meta.tsv"
CONLLU_EXT = ".conllu"

PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = "ParlaParla"
WANDB_ID = generate_id()
