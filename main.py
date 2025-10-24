# import argparse
import os
from omegaconf import OmegaConf
from dataset.parliment_dataset import ParlimentDataset

# parser = argparse.ArgumentParser(description="ParlaParla Project")


if __name__ == "__main__":

    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")

    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    dataset = ParlimentDataset(config)
    print(len(dataset))
