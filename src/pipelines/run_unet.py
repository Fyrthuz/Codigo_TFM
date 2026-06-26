"""Entry point for running the 2D UNet pipeline."""
import argparse
import os

from src.config import PipelineConfig
from src.pipelines.unet import UNetPipeline


def main():
    parser = argparse.ArgumentParser(description="UNet 2D segmentation + uncertainty pipeline")
    parser.add_argument("--config", type=str, default="./configs/pipeline_2d.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to UNet checkpoint (optional — trains if missing)")
    parser.add_argument("--train", action="store_true",
                        help="Train a model first, then run the pipeline")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (only if --train or no checkpoint)")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)

    if args.train or (args.checkpoint and not os.path.exists(args.checkpoint)):
        print("Training UNet...")
        from src.utils.train_unet import main as train_main
        import sys
        sys.argv = [
            "train_unet",
            "--data-root", config.paths.get("root_path", "./MRI/filtered_data"),
            "--epochs", str(args.epochs),
            "--save-path", args.checkpoint or "unet_model.pth",
        ]
        train_main()
        args.checkpoint = args.checkpoint or "unet_model.pth"

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        available = [f for f in os.listdir(".") if f.endswith(".pth") or f.endswith(".pt")]
        if available:
            args.checkpoint = available[0]
            print(f"Using checkpoint: {args.checkpoint}")
        else:
            print("No checkpoint found. Training a model first...")
            from src.utils.train_unet import main as train_main
            import sys
            sys.argv = ["train_unet", "--data-root", config.paths.get("root_path", "./MRI/filtered_data")]
            train_main()
            args.checkpoint = "unet_model.pth"

    pipeline = UNetPipeline.from_yaml(args.config, checkpoint_path=args.checkpoint)
    pipeline.run()


if __name__ == "__main__":
    main()
