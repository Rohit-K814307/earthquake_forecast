import torch
import torch.optim as optim
from torch.nn import MSELoss, L1Loss
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from eq_forecast.visualization.heatmap import make_pred_heatmaps
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_and_eval(train_set,
                   test_set,
                   val_set,
                   grid,
                   model,
                   num_epochs,
                   lr,
                   lr_factor,
                   lr_patience,
                   model_name,
                   batch_size,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                   checkpoint_dir="checkpoints",
                   log_dir="runs",
                   save_interval=5,
                   early_stop_patience=10):
    
    # Make necessary dirs
    os.makedirs(f"eq_forecast/models/{model_name}/{checkpoint_dir}", exist_ok=True)
    os.makedirs(f"eq_forecast/models/{model_name}/{log_dir}", exist_ok=True)
    os.makedirs(f"eq_forecast/models/{model_name}/heatmaps", exist_ok=True)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=lr_factor, patience=lr_patience, verbose=True)
    criterion = MSELoss()

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    # Metrics
    mse_metric = MSELoss()
    mae_metric = L1Loss()


    writer = SummaryWriter(f"eq_forecast/models/{model_name}/{log_dir}")

    model.to(device)
    print(f"Training {model_name} model:")
    print(model)

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        y_true_train = []
        y_pred_train = []

        train_progress = tqdm(train_set, desc=f"Epoch {epoch+1}/{num_epochs} | Training", total=len(train_set.features),unit="batch")

        for data in train_progress:
            features = data.x.to(device)
            labels = data.y.to(device)
            pad_matrix = data.y_pad.to(device).float()
            edge_idx = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            
            optimizer.zero_grad()
            predictions = model(features, edge_idx, edge_attr)

            # Compute loss with pad mask
            pad_matrix = pad_matrix.unsqueeze(1).unsqueeze(2)
            loss = criterion(predictions * pad_matrix, labels * pad_matrix)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size  # Scale by batch size

            # Collect predictions
            y_true_train.append(labels.cpu())
            y_pred_train.append(predictions.cpu().detach())

        # Compute training metrics
        train_loss /= len(train_set.features)  # Normalize by dataset size

        train_mse = mse_metric(torch.stack(y_true_train), torch.stack(y_pred_train))
        train_rmse = np.sqrt(train_mse)
        train_mae = mae_metric(torch.stack(y_true_train), torch.stack(y_pred_train))

        # Validation loop
        model.eval()
        val_loss = 0
        y_true_val = []
        y_pred_val = []

        val_progress = tqdm(val_set, desc=f"Epoch {epoch+1}/{num_epochs} | Validation", total=len(val_set.features), unit="batch")

        with torch.no_grad():
            val_step = 0
            for data in val_progress:
                val_step += 1
                features = data.x.to(device)
                labels = data.y.to(device)
                edge_idx = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)


                predictions = model(features, edge_idx, edge_attr)
                loss = criterion(predictions, labels)
                val_loss += loss.item() * batch_size  # Scale by batch size

                if val_step % 5 == 0:
                    make_pred_heatmaps(predictions, labels, grid, f"eq_forecast/models/{model_name}/heatmaps/epoch_{epoch}_valstep_{val_step}_mag_pred.html",f"eq_forecast/models/{model_name}/heatmaps/epoch_{epoch}_valstep_{val_step}_mag_act.html","Magnitude", 0)
                    make_pred_heatmaps(predictions, labels, grid, f"eq_forecast/models/{model_name}/heatmaps/epoch_{epoch}_valstep_{val_step}_depth_pred.html",f"eq_forecast/models/{model_name}/heatmaps/epoch_{epoch}_valstep_{val_step}_depth_act.html","Depth", 1)
                
                # Collect predictions and labels
                y_true_val.append(labels.cpu())
                y_pred_val.append(predictions.cpu())

        # Compute validation metrics
        val_loss /= len(val_set.features)

        val_mse = mse_metric(torch.stack(y_true_val), torch.stack(y_pred_val))
        val_rmse = np.sqrt(val_mse)
        val_mae = mae_metric(torch.stack(y_true_val), torch.stack(y_pred_val))

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping...")
                break

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"eq_forecast/models/{model_name}/{checkpoint_dir}/model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    # Eval
    model.load_state_dict(best_model)
    model.eval()
    test_loss = 0
    y_true_test = []
    y_pred_test = []

    with torch.no_grad():
        for data in test_set:
            features = data.x.to(device)
            labels = data.y.to(device)
            edge_idx = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)

            predictions = model(features, edge_idx, edge_attr)
            loss = criterion(predictions, labels)
            test_loss += loss.item() * batch_size

            y_true_test.append(labels.cpu())
            y_pred_test.append(predictions.cpu())

    test_loss /= len(test_set.features)

    test_mse = mse_metric(torch.stack(y_true_test), torch.stack(y_pred_test))
    test_rmse = np.sqrt(test_mse)
    test_mae = mae_metric(torch.stack(y_true_test), torch.stack(y_pred_test))

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}')

    torch.save(best_model, f"eq_forecast/models/{model_name}/{checkpoint_dir}/best_model.pth")
    print(f"Best model saved!")

    writer.close()
