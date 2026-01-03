import torch
from argparse import ArgumentParser


from layers.BlockLinear import BlockLinear
from layers.cacheAttention import AttentionOptimized

def main(argc: int, *argv: str) -> int:
    # parser = ArgumentParser()
    # parser.add_argument("--configPath", type=str, default="config.json")
    # parser.parse_args(argv)

    return 0

if __name__ == '__main__':
    argv = __import__("sys").argv
    exit(main(len(argv), *argv))