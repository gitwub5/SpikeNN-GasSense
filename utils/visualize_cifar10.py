import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so we can import from other directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import CIFAR10_MODEL_WEIGHTS_DIR, CIFAR10_ANALYZE_DIR
from dataset.cifar10_dataset import get_cifar10_dataloaders
from model.cifar10_spiking_net import CIFAR10SpikingNet

# CIFAR-10 classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def unnormalize(img):
    """Unnormalize an image back to [0, 1] range for plotting."""
    # Mean and Std from dataset
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(-1, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(-1, 1, 1)
    
    img = img.numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0)) # Change to (H, W, C)

def plot_conv_weights(model, save_path):
    """Plot the weights of the first convolutional layer."""
    print("Plotting first Conv layer weights...")
    weights = model.conv1.weight.data.cpu()
    
    # We will plot the first 32 filters
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(32, weights.shape[0])):
        w = weights[i].numpy()
        # Normalize weight to [0, 1] for visualization
        w_min, w_max = w.min(), w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-5)
        # Transpose from (C, H, W) to (H, W, C)
        w_norm = np.transpose(w_norm, (1, 2, 0))
        
        axes[i].imshow(w_norm)
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Weights visualization saved to: {save_path}")

def plot_inference_results(model, test_loader, device, num_steps, save_path):
    """Plot inference results for a batch of test data."""
    print("Running inference and plotting results...")
    model.eval()
    
    # Get a single batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # We only take the first 16 images for visualization
    images = images[:16]
    labels = labels[:16]
    
    images_device = images.to(device)
    
    with torch.no_grad():
        spk_rec, _ = model(images_device, num_steps)
        # Sum spikes over time, and take the argmax to get the predicted class
        _, predicted = spk_rec.sum(dim=0).max(1)
        predicted = predicted.cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(16):
        ax = axes[i]
        img = unnormalize(images[i])
        true_label = CLASSES[labels[i]]
        pred_label = CLASSES[predicted[i]]
        
        ax.imshow(img)
        ax.axis('off')
        
        # Color the text red if prediction is wrong, green if correct
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, color=color, fontsize=10)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('CIFAR-10 SNN Inference Results', fontsize=16)
    plt.savefig(save_path)
    plt.close()
    print(f"Inference visualization saved to: {save_path}")

def main():
    # Ensure analysis directory exists
    CIFAR10_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model_path = CIFAR10_MODEL_WEIGHTS_DIR / "best_cifar10_snn.pth"
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        sys.exit(1)
        
    net = CIFAR10SpikingNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Load Dataloader
    _, test_loader = get_cifar10_dataloaders(batch_size=128)
    
    # 1. Visualize Weights
    weight_save_path = CIFAR10_ANALYZE_DIR / "conv1_weights.png"
    plot_conv_weights(net, weight_save_path)
    
    # 2. Visualize Inference
    inference_save_path = CIFAR10_ANALYZE_DIR / "inference_results.png"
    plot_inference_results(net, test_loader, device, num_steps=25, save_path=inference_save_path)

if __name__ == "__main__":
    main()
