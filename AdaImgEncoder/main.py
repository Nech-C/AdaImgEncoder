import torch
from model.image_encoder import CustomImageEncoder
from data.dataset import load_and_process_data
from utils.helpers import create_projector, get_optimizer_and_scheduler
from training.trainer import train

def main():
    # Hyperparameters
    d_model = 128
    vision_dim = 1024
    clip_dim = 768
    max_length = 6
    num_epochs = 10
    batch_size = 64
    learning_rate = 5e-4
    warmup_steps = 1000

    # Load data
    train_loader, val_loader, test_loader = load_and_process_data(num_samples=10000, batch_size=batch_size)

    # Initialize model and projector
    model = CustomImageEncoder(d_model, vision_dim, 8, 512, 0.05, 6, None, max_length)
    projector = create_projector(d_model, clip_dim)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    projector.to(device)

    # Create optimizer and scheduler
    total_steps = num_epochs * len(train_loader)
    optimizer, scheduler = get_optimizer_and_scheduler(model, projector, learning_rate, warmup_steps, total_steps)

    # Train the model
    train(model, projector, train_loader, val_loader, optimizer, scheduler, num_epochs, device, max_length)

    print("Training completed.")

if __name__ == "__main__":
    main()