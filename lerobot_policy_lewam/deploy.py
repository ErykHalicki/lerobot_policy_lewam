"""
Pull latest checkpoint from S3, export to HuggingFace, and clear local cache.

Usage:
    python -m lerobot_policy_lewam.deploy cube_single_task ehalicki/lewam-so101-cube
    python -m lerobot_policy_lewam.deploy cube_single_task ehalicki/lewam-so101-cube --cameras image1
"""

import argparse
import os
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag", help="S3 run tag (e.g. cube_single_task)")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. ehalicki/lewam-so101-cube)")
    parser.add_argument("--s3-path", default="s3://lewam/checkpoints")
    parser.add_argument("--cameras", nargs="+", default=["image1", "image2"])
    args = parser.parse_args()

    s3_key = f"{args.s3_path}/{args.run_tag}/{args.run_tag}_latest.pt"
    local_ckpt = os.path.join(tempfile.gettempdir(), f"{args.run_tag}_latest.pt")

    print(f"Pulling {s3_key} ...")
    subprocess.run(["aws", "s3", "cp", s3_key, local_ckpt], check=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nExporting + pushing to {args.repo_id} ...")
        from lerobot_policy_lewam.export_to_hf import export
        export(local_ckpt, tmp_dir, args.cameras, repo_id=args.repo_id)

    os.remove(local_ckpt)

    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        f"models--{args.repo_id.replace('/', '--')}",
    )
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared HF cache: {cache_dir}")

    print(f"\nDone. Serve with:")
    print(f"  python -m lerobot_policy_lewam.serve_lewam --model {args.repo_id} --device cuda")


if __name__ == "__main__":
    main()
