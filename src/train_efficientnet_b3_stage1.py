import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

from src.data.data_loader import load_datasets
from src.config.device_config import configure_device

def build_efficientnet_b3(num_classes):
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def main():
    configure_device()

    train_ds, valid_ds, class_names = load_datasets()
    num_classes = len(class_names)

    model = build_efficientnet_b3(num_classes)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=3
    )

    EXP_DIR = Path("experiments/efficientnet_b3/stage1_head_training")
    VIS_DIR = EXP_DIR / "visualizations"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)

    with open(EXP_DIR / "history.json", "w") as f:
        json.dump(history.history, f, indent=4)

    model.save("models/efficientnet_b3_stage1.h5")

    print("EfficientNet-B3 Stage 1 training complete")


if __name__ == "__main__":
    main()
