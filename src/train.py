import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device, pad_idx):
    model.train()
    total_loss = 0
    
    for batch_idx, (imgs, captions) in enumerate(tqdm(dataloader, desc="Training")):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        # Target is shifted by 1 (we want to predict the next word)
        targets = captions[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass (model handles shifting captions for teacher forcing)
        outputs = model(imgs, captions)
        
        # Flatten outputs and targets for cross entropy
        # outputs shape: (batch_size, seq_len - 1, vocab_size)
        # targets shape: (batch_size, seq_len - 1)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, captions in tqdm(dataloader, desc="Validating"):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            targets = captions[:, 1:]
            
            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, vocab, num_epochs=10, learning_rate=3e-4, device="cuda"):
    model = model.to(device)
    
    pad_idx = vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, pad_idx)
        val_loss = validate_epoch(model, val_loader, criterion, device, pad_idx)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved new best model.")
            
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig("checkpoints/training_curve.png")
    plt.close()
    
    return train_losses, val_losses
