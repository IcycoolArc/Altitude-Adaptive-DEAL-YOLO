import os, warnings, re, multiprocessing as mp

# Make sure child worker processes inherit the warning filter
os.environ["PYTHONWARNINGS"] = "ignore:.*does not have a deterministic implementation.*"
# Force PyTorch non-deterministic path even if something flips it later
os.environ["TORCH_DETERMINISTIC"] = "0"

# Hide those deterministic warnings from PyTorch
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning
)

def main():
    import torch
    # Hard-disable determinism
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from ultralytics import YOLO
    m = YOLO(r"runs/train/yolov8m-ASF-P212/weights/last.pt")

    # Force overrides to win over checkpoint args
    m.train(
        resume=True,
        data=r"C:/Users/Arc/Experiment-YOLO/datasets/WAID/data.yaml",
        project=r"runs/train",
        name="yolov8m-ASF-P212",
        exist_ok=True,
        deterministic=False,   # <-- force it OFF
        workers=0,             # <-- stable on Windows; increase later if you want
    )

if __name__ == "__main__":
    mp.freeze_support()  # important on Windows
    main()
