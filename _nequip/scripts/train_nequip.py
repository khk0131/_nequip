#!/usr/bin/env python

import argparse
import pathlib
import yaml

from _nequip import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='モデルをtrainする'
        )
    parser.add_argument('train_config_path', help='trainのconfig')

    args = parser.parse_args()

    train_config_path = pathlib.Path(args.train_config_path)
    
    with open(train_config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()