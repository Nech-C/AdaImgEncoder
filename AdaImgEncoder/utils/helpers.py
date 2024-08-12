import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

def create_projector(d_model, clip_dim):
    return nn.Sequential(
        nn.Linear(d_model, clip_dim),
        nn.LayerNorm(clip_dim),
        nn.ReLU(),
        nn.Linear(clip_dim, clip_dim)
    )

def get_optimizer_and_scheduler(model, projector, lr, num_warmup_steps, num_training_steps):
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def save_model(model, projector, optimizer, epoch, train_loss, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)