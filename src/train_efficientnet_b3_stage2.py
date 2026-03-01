import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data.data_loader import load_datasets
from src.config.device_config import configure_device


def unfreeze_top_layers(model, unfreeze_ratio=0.4):
    
    total_layers = len(model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_ratio))

    for layer in model.layers[unfreeze_from:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    print(f"Unfroze top {int(unfreeze_ratio*100)}% layers")


def main():
    configure_device()

    # Load data
    train_ds, valid_ds, class_names = load_datasets()

    # Load Stage-1 model
    model = tf.keras.models.load_model(
        "models/efficientnet_b3_stage1.h5"
    )

    unfreeze_top_layers(model, unfreeze_ratio=0.4)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    EXP_DIR = Path("experiments/efficientnet_b3/stage2_finetuning")
    VIS_DIR = EXP_DIR / "visualizations"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="models/efficientnet_b3_stage2.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15,
        callbacks=callbacks
    )

    with open(EXP_DIR / "history.json", "w") as f:
        json.dump(history.history, f, indent=4)

    print("EfficientNet-B3 Stage-2 fine-tuning complete")


if __name__ == "__main__":
    main()
