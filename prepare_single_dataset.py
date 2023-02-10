import argparse
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser(description='Load dataset and store it in tfds cache.')
parser.add_argument('dataset', type=str, help='dataset name to load')

args = parser.parse_args()

dataset = tfds.builder(args.dataset)
dataset.download_and_prepare()