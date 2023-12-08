#!/usr/bin/env python3

import argparse
import pathlib
import yaml

from _nequip import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='モデルをevalする'
        )
    parser.add_argument('eval_config_path', help='evalのconfig')

    args = parser.parse_args()

    eval_config_path = pathlib.Path(args.eval_config_path)
    
    with open(eval_config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config)
    evaluator.eval()