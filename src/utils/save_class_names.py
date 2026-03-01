import json
from pathlib import Path
import tensorflow as tf

DATASET_ROOT = Path("dataset/New Plant Diseases Dataset(Augmented)/train")

OUTPUT_PATH = Path("models/class_names.json")

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_ROOT}")

    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_ROOT,
        labels="inferred",
        label_mode="int",
        image_size=(224, 224),
        shuffle=False
    )

    class_names = ds.class_names

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    print("class_names.json saved successfully")
    print(f"Number of classes: {len(class_names)}")
    print("First 5 classes:", class_names[:5])

if __name__ == "__main__":
    main()
