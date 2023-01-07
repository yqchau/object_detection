import numpy as np
import torch
from tqdm import tqdm

from datasets import SampleHeatMapDataset
from model import CNN18_Backbone

if __name__ == "__main__":
    torch.manual_seed(0)
    train_dataset = SampleHeatMapDataset(path="./dataset/5_XYHW/train")
    val_dataset = SampleHeatMapDataset(path="./dataset/5_XYHW/val")
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-4
    n_epochs = 100
    n_workers = 8
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size
        # , num_workers=n_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
    )
    model = CNN18_Backbone(3)  # CNN18(3, 20)  # FCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_loss_value = -1
    avg_loss = -1
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []

        progress_bar = tqdm(train_loader, leave=False)
        for (x, y) in progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()
            y_pred = model(x)

            # loss = sigmoid_focal_loss(y_pred, y, reduction="mean")
            loss = torch.nn.functional.mse_loss(torch.sigmoid(y_pred), y)
            loss_value = loss.item()
            epoch_loss.append(loss_value)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=loss_value, mean_loss=avg_loss, val_loss=val_loss_value
            )

        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            val_progress_bar = tqdm(val_loader, leave=False)
            for (x, y) in val_progress_bar:
                val_progress_bar.set_description(f"Validation Epoch {epoch}")
                y_pred = model(x)
                # loss = sigmoid_focal_loss(y_pred, y, reduction="mean")
                loss = torch.nn.functional.mse_loss(torch.sigmoid(y_pred), y)
                val_epoch_loss.append(loss.item())

        val_loss_value = np.mean(val_epoch_loss)
        avg_loss = np.mean(epoch_loss)

        # print(
        #     f"Epoch: {epoch}, Loss: {np.mean(epoch_loss):.6f}, Val_Loss: {np.mean(val_epoch_loss):.6f}"
        # )

        if not ((epoch + 1) % 1):
            torch.save(model.state_dict(), "./weights.pth")
