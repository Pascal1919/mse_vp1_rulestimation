import os
import wandb

import torch
import numpy as np


class Trainer:
    def __init__(self, model, model_optimizer, print_every, max_rul=125, epochs=200, device='cuda', prefix='13', handcrafted=False):
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.prefix = prefix
        self.max_rul = max_rul
        self.model_name = wandb.run.config["model"]
        self.handcrafted = handcrafted

    def train_single_epoch(self, dataloader, global_step):
        self.model.train()
        running_loss = 0
        total = 0.0
        for batch_index, data in enumerate(dataloader, 0):
            X, handcrafted_feature, y = data
            X, handcrafted_feature, y = X.to(self.device), handcrafted_feature.to(self.device), y.to(self.device)
            self.model_optimizer.zero_grad()
            if self.handcrafted:
                predictions = self.model(X, handcrafted_feature)
            else:
                predictions = self.model(X)
            loss = self.criterion(y, predictions.squeeze(1))
            loss.backward()
            self.model_optimizer.step()

            running_loss += loss.item() * X.size(0)
            total += X.size(0)
            wandb.log({
                "train/loss": loss.item(),
                "train/RMSE": loss.item()**0.5
                }, step=global_step)

        train_loss = (running_loss / total)
        train_rmse = train_loss**0.5
        return train_loss, train_rmse, global_step
    
    def test(self, test_loader, global_step):
        self.model.eval()
        running_loss = 0.0
        running_score = 0.0
        running_loss_rul = 0.0
        running_score_rul = 0.0
        total = 0.0
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for batch_index, data in enumerate(test_loader, 0):
                X, handcrafted_feature, y = data
                X, handcrafted_feature, y = X.to(self.device), handcrafted_feature.to(self.device), y.to(self.device)
                if self.handcrafted:
                    predictions = self.model(X, handcrafted_feature)
                else:
                    predictions = self.model(X)
                score_rul = self.score(y * self.max_rul, predictions.squeeze(1) * self.max_rul)
                loss_rul = criterion(y * self.max_rul, predictions.squeeze(1) * self.max_rul)

                score = self.score(y, predictions.squeeze(1))
                loss = criterion(y, predictions.squeeze(1))

                running_loss += loss.item() * X.size(0)
                running_score += score.item() * X.size(0)
                running_loss_rul += loss_rul.item() * X.size(0)
                running_score_rul += score_rul.item() * X.size(0)

                total += X.size(0)

        val_loss = running_loss / total
        val_score = running_score / total
        val_loss_rul = running_loss_rul / total
        val_score_rul = running_score_rul / total
        val_rmse = val_loss**0.5
        val_rmse_rul = val_loss_rul**0.5

        wandb.log({
            "validation/loss": val_loss,
            "validation/RMSE": val_rmse,
            "validation/score": val_score,
            "validation/loss_rul": val_loss_rul,
            "validation/RMSE_rul": val_rmse_rul,
            "validation/score_rul": val_score_rul
        }, step=global_step)
        global_step += 1

        print('test result: score: {}, RMSE: {}'.format(val_score, val_rmse))
        return val_loss, val_rmse, val_score, global_step

    def train(self, train_loader, test_loader, iteration, global_step):
        train_step = global_step
        val_step = global_step
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            
            train_loss, train_rmse, global_step = self.train_single_epoch(train_loader, global_step)
            val_loss, val_rmse, val_score, global_step = self.test(test_loader, global_step)
            if epoch == 0:
                best_score = val_score
                best_RMSE = val_rmse
            else:
                if val_score < best_score:
                    best_score = val_score
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_score')
                if val_rmse < best_RMSE:
                    best_RMSE = val_rmse
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_RMSE')
            # print(f"epoch={epoch}, loss={current_RMSE}, score={current_score}")
            wandb.log({
                "epoch/epoch": epoch + 1,
                "epoch/train-loss": train_loss,
                "epoch/train-RMSE": train_rmse,
                "epoch/val-loss": val_loss,
                "epoch/val-RMSE": val_rmse,
                "epoch/val-score": val_score
            }, step=global_step)
        return float(best_score), float(best_RMSE), global_step

    def save_checkpoints(self, iteration, epoch, which_type):
        save_dir = './checkpoints/{}/'.format(self.model_name)
        save_path = os.path.join(save_dir, '{}_iteration{}_{}.pth.tar'.format(self.prefix, iteration, which_type))

        os.makedirs(save_dir, exist_ok=True)

        state = {
            'iter': iteration,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.model_optimizer.state_dict()
        }
        torch.save(state, save_path)
        print(f'{which_type}_checkpoints saved successfully at {save_path}!')

    @staticmethod
    def score(y_true, y_pred):
        score = 0
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        for i in range(len(y_pred)):
            if y_true[i] <= y_pred[i]:
                score = score + np.exp(-(y_true[i] - y_pred[i]) / 10.0) - 1
            else:
                score = score + np.exp((y_true[i] - y_pred[i]) / 13.0) - 1
        return score