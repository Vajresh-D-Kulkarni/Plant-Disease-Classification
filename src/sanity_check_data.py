from src.config.device_config import configure_device
configure_device(cpu_only=True)

from src.data.data_loader import load_datasets

def main():
    train_ds, valid_ds, class_names = load_datasets()

    print("Number of classes:", len(class_names))
    print("Class names (first 10):", class_names[:10])

    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)

if __name__ == "__main__":
    main()
