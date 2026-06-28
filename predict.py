from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "ultralytics" / "mycfg" / "yolov11n-face.pt"
SOURCE = ROOT / "ultralytics" / "assets" / "face.jpg"


def main():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install project dependencies first, "
            "for example: pip install -e ."
        ) from exc

    model = YOLO(str(WEIGHTS))

    results = model.predict(
        source=str(SOURCE),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(ROOT / "runs" / "predict"),
        name="face",
        exist_ok=True,
    )

    for result in results:
        print(f"image: {result.path}")
        print(f"save_dir: {result.save_dir}")
        if result.boxes is None:
            print("detections: 0")
            continue
        print(f"detections: {len(result.boxes)}")


if __name__ == "__main__":
    main()
