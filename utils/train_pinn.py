import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.pinn import Score


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        # self.global_step = 0

    def train_epoch(self, train_loader, global_step):
        """Führt eine Trainings-Epoche aus."""
        self.model.train()
        running_loss, running_data_loss, running_physics_loss, running_score = 0.0, 0.0, 0.0, 0.0
        total = 0.0

        for X, y, t in train_loader:
            X, t, y = X.to(self.device), t.to(self.device).view(-1, 1), y.to(self.device).view(-1, 1)
            self.optimizer.zero_grad()
            u, f = self.model(X, t)
            loss, data_loss, physics_loss = self.criterion(u, f, y)
            loss.backward()
            self.optimizer.step()

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            running_data_loss += data_loss.item() * batch_size
            running_physics_loss += physics_loss.item() * batch_size
            score = Score.compute_score(u, y)
            running_score += score.item() * batch_size
            total += batch_size

            wandb.log({
                "train/loss": loss.item(),
                "train/data-loss": data_loss.item(),
                "train/physics-loss": physics_loss.item(),
                "train/score": score.item()
            }, step=global_step)
            

        return self._compute_epoch_metrics(running_loss, running_data_loss, running_physics_loss, running_score, total, global_step)

    def validation_epoch(self, val_loader, global_step):
        """Führt eine Validierungs-Epoche aus."""
        self.model.eval()
        running_loss, running_data_loss, running_physics_loss, running_score = 0.0, 0.0, 0.0, 0.0
        total = 0.0

        for X, y, t in val_loader:
            X, t, y = X.to(self.device), t.to(self.device).view(-1, 1), y.to(self.device).view(-1, 1)
            u, f = self.model(X, t)
            loss, data_loss, physics_loss = self.criterion(u, f, y)
            score = Score.compute_score(u, y)

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            running_data_loss += data_loss.item() * batch_size
            running_physics_loss += physics_loss.item() * batch_size
            running_score += score.item() * batch_size
            total += batch_size

        global_step += 1    

        return self._compute_epoch_metrics(running_loss, running_data_loss, running_physics_loss, running_score, total, global_step)

    def fit(self):
        """Trainiert das Modell über mehrere Epochen."""
        global_step = 0
        for epoch in range(self.config["num_epochs"]):
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")

            train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Training Epoch {epoch+1}')
            val_loader = tqdm(self.val_loader, total=len(self.val_loader), desc=f'Validation Epoch {epoch+1}')

            train_loss, train_data_loss, train_physics_loss, train_score, global_step = self.train_epoch(train_loader, global_step)
            val_loss, val_data_loss, val_physics_loss, val_score, global_step = self.validation_epoch(val_loader, global_step)

            wandb.log({
                "epoch/epoch": epoch + 1,
                "epoch/train-loss": train_loss,
                "epoch/train-data-loss": train_data_loss,
                "epoch/train-physics-loss": train_physics_loss,
                "epoch/train-score": train_score,
                "epoch/val-loss": val_loss,
                "epoch/val-data-loss": val_data_loss,
                "epoch/val-physics-loss": val_physics_loss,
                "epoch/val-score": val_score
            }, step=global_step)

        

    def _compute_epoch_metrics(self, loss, data_loss, physics_loss, score, total, global_step):
        """Hilfsfunktion zur Berechnung der durchschnittlichen Verluste und Score-Werte."""
        epoch_loss = loss / total
        epoch_data_loss = data_loss / total
        epoch_physics_loss = physics_loss / total
        epoch_score = score / total
        # self.global_step += 1
        return epoch_loss, epoch_data_loss, epoch_physics_loss, epoch_score, global_step

def test_model(model, ds_test, batch_size, device='cuda'):

    
    test_loader = DataLoader(ds_test, batch_size, shuffle=False, num_workers=0)
    # Move model to evaluation mode
    model.eval()
    model.to(device)

    all_predictions = []
    all_ground_truth = []
    test_loss = 0.0
    test_score = 0.0

    # Define loss function
    criterion = torch.nn.MSELoss()

    for X, y, t in tqdm(test_loader, desc="Testing"):
        # Move data to device
        X, y, t = X.to(device), y.to(device).view(-1, 1), t.to(device).view(-1, 1)

        # Forward pass
        predictions, _ = model(X, t)

        # Compute loss
        loss = criterion(predictions, y)**0.5
        score = Score.compute_score(predictions, y)
        test_loss += loss.item() * X.size(0)
        test_score += score.item() * X.size(0)

        # Collect predictions and ground truth
        all_predictions.append(predictions.cpu())
        all_ground_truth.append(y.cpu())

    # Compute average loss
    test_loss /= len(ds_test)
    test_score /= len(ds_test)

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)

    print(f"Test RMSE: {test_loss:.4f}, Test Score: {test_score:.4f}")
    return test_loss, test_score, all_predictions, all_ground_truth