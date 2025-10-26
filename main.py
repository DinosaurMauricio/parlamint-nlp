import os
from omegaconf import OmegaConf
from functools import partial
from utils.data import load_data, DataPipeline
from dataset.parliment import ParlimentDataset


if __name__ == "__main__":

    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")

    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    print("Loading dataset...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    pipeline = DataPipeline(config)
    data = pipeline.prepare_dataset(raw_data)
