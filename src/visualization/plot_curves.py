import matplotlib.pyplot as plt
from src.visualization.utils import load_history, ensure_dir

def plot_training_curves(history_path, output_dir):
    history = load_history(history_path)
    ensure_dir(output_dir)

    epochs = range(1, len(history["loss"]) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_curve.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.close()
