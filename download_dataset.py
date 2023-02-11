import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from flan.v2 import mixtures
import seqio

import json
from tqdm import tqdm
import argparse

MAX_POSSIBLE_SAMPLES = int(1e12)

parser = argparse.ArgumentParser(description='Load dataset and save as json')
parser.add_argument('--task-or-mixture', type=str, default="flan_zs_fs_opt", help='dataset or mixture to load')
parser.add_argument('--output-path', type=str, help='output directory')
parser.add_argument('--num-examples', type=str, default="all", help='number of examples to load')

args = parser.parse_args()

if args.num_examples == "all":
    args.num_examples = MAX_POSSIBLE_SAMPLES
else:
    args.num_examples = int(args.num_examples)

if args.output_path is None:
    args.output_path = "./" + args.task_or_mixture + ".jsonl"

if args.task_or_mixture == "flan_zs_fs_opt":
    seqio.MixtureRegistry.add(
    'flan_zs_fs_opt',
    tasks=[
        ('flan_zsopt', 50),  # mixing weight = 50
        ('flan_fsopt', 50),  # mixing weight = 50
    ])

task = seqio.get_mixture_or_task(args.task_or_mixture)
num_input_examples = task.num_input_examples("train")
args.num_examples = min(args.num_examples, num_input_examples)

dataset = seqio.get_mixture_or_task(args.task_or_mixture).get_dataset(
        sequence_length={'inputs':4096,'targets':4096}, # Extranous length to capture all data
        num_epochs=1,
        copy_pretokenized=True,
        shuffle=True,
        passthrough_features=["task_name"],
)

with open(args.output_path, "w") as f:
    for i, example in tqdm(enumerate(dataset.as_numpy_iterator()), total=args.num_examples):
        raw_input_sequence = example['inputs_pretokenized'].decode()
        raw_target_sequence = example['targets_pretokenized'].decode()
        json_string = json.dumps(
            {"inputs": raw_input_sequence,
             "targets": raw_target_sequence,
             "input_seq_len": len(example['inputs']),
             "target_seq_len": len(example['targets'])})
        f.write(json_string)
        f.write("\n")
        if i >= args.num_examples - 1:
            break