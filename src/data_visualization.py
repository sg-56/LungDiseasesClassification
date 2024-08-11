import os
import matplotlib.pyplot as plt
import numpy as np

def plot_image_distribution(base_path, class_names):
    """Plot distribution of images in train, val, and test sets."""
    class_counts = {class_name: {"train": 0, "val": 0, "test": 0} for class_name in class_names}

    for subset in ["train", "test", "val"]:
        subset_path = os.path.join(base_path, subset)
        for class_name in class_names:
            class_path = os.path.join(subset_path, class_name)
            num_images = len(os.listdir(class_path))
            class_counts[class_name][subset] = num_images

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']

    for i, subset in enumerate(["train", "val", "test"]):
        classes = list(class_counts.keys())
        counts = [class_counts[class_name][subset] for class_name in classes]
        bars = axs[i].bar(classes, counts, color=colors)
        axs[i].set_title(f'Distribution of Images in {subset.capitalize()} Set')
        axs[i].set_xlabel('Class')
        axs[i].set_ylabel('Number of Images')
        for bar in bars:
            yval = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2, yval, round(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training and validation loss and accuracy."""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
