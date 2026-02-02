"""Split a large safetensors file into multiple shards for faster parallel loading."""

import argparse
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

def split_safetensors(input_path: Path, output_dir: Path, num_shards: int):
    """Split a safetensors file into multiple shards.

    Args:
        input_path: Path to the input .safetensors file
        output_dir: Directory to save the sharded files
        num_shards: Number of shards to create
    """
    print(f"Loading model from {input_path}...")
    state_dict = load_file(str(input_path))

    # Get all parameter names
    param_names = list(state_dict.keys())
    print(f"Found {len(param_names)} parameters")

    # Calculate total size
    total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"Total model size: {total_size / (1024**3):.2f} GB")

    # Sort parameters by size (largest first) for better load balancing
    param_sizes = [(name, state_dict[name].numel() * state_dict[name].element_size())
                   for name in param_names]
    param_sizes.sort(key=lambda x: x[1], reverse=True)

    # Distribute parameters across shards using greedy algorithm
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards

    for param_name, param_size in param_sizes:
        # Add to the smallest shard
        min_idx = shard_sizes.index(min(shard_sizes))
        shards[min_idx].append(param_name)
        shard_sizes[min_idx] += param_size

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each shard
    weight_map = {}
    for shard_idx, param_names_in_shard in enumerate(shards):
        shard_filename = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = output_dir / shard_filename

        print(f"\nShard {shard_idx+1}/{num_shards}: {len(param_names_in_shard)} parameters, "
              f"{shard_sizes[shard_idx] / (1024**3):.2f} GB")

        # Create shard dict
        shard_dict = {name: state_dict[name] for name in param_names_in_shard}

        # Save shard
        print(f"  Saving to {shard_path}...")
        save_file(shard_dict, str(shard_path))

        # Update weight map
        for name in param_names_in_shard:
            weight_map[name] = shard_filename

    # Create index file
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }

    index_path = output_dir / "model.safetensors.index.json"
    print(f"\nSaving index to {index_path}...")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\nDone! Created {num_shards} shards in {output_dir}")
    print(f"Average shard size: {sum(shard_sizes) / num_shards / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Split safetensors file into shards")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .safetensors file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save sharded files"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=8,
        help="Number of shards to create (default: 8)"
    )

    args = parser.parse_args()

    split_safetensors(args.input, args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
