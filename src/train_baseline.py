from src.config.device_config import configure_device
configure_device(cpu_only=True)
import json

from src.data.data_loader import load_datasets
from src.models.cnn_model import build_baseline_cnn

EPOCHS = 10

def main():
    train_ds, valid_ds, class_names = load_datasets()

    print(f"Training baseline CNN with {len(class_names)} classes")

    model = build_baseline_cnn(num_classes=len(class_names))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS
    )

    model.save("models/baseline_cnn.h5")
    print("Baseline model saved to models/baseline_cnn.h5")

    with open("models/baseline_history.json", "w") as f:
        json.dump(history.history, f, indent=4)


    return history

if __name__ == "__main__":
    main()
