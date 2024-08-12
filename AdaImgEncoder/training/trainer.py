import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from ..utils.helpers import save_model

def train_epoch(model, projector, train_loader, optimizer, scheduler, device, max_length):
    model.train()
    projector.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        image_embed = batch["image_embed"].to(device)
        text_embeds = batch["text_embed"].to(device)

        generated_sequence = model.generate(image_embed, max_length)

        loss = 0
        for i in range(1, generated_sequence.size(1)):
            projected_token = projector(generated_sequence[:, i, :])
            for j in range(5):
                similarity = F.cosine_similarity(projected_token, text_embeds[:, j, :], dim=1)
                loss -= similarity.mean() * i / (max_length - 1)

        loss /= 5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, projector, val_loader, device, max_length):
    model.eval()
    projector.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            image_embed = batch["image_embed"].to(device)
            text_embeds = batch["text_embed"].to(device)

            generated_sequence = model.generate(image_embed, max_length)

            loss = 0
            for i in range(1, generated_sequence.size(1)):
                projected_token = projector(generated_sequence[:, i, :])
                for j in range(5):
                    similarity = F.cosine_similarity(projected_token, text_embeds[:, j, :], dim=1)
                    loss -= similarity.mean() * i / (max_length - 1)

            loss /= 5
            total_loss += loss.item()

    return total_loss / len(val_loader)

def train(model, projector, train_loader, val_loader, optimizer, scheduler, num_epochs, device, max_length):
    wandb.init(project="image-encoding-project", name="experiment-1")
    wandb.config.update({
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "initial_learning_rate": optimizer.param_groups[0]['lr'],
        "model": model.__class__.__name__,
        "max_length": max_length
    })

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, projector, train_loader, optimizer, scheduler, device, max_length)
        val_loss = validate(model, projector, val_loader, device, max_length)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, projector, optimizer, epoch, train_loss, val_loss, 'best_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    wandb.finish()