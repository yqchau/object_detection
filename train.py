import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from datasets import SampleDataset
from model import CNN18

if __name__ == "__main__":
    torch.manual_seed(0)
    image_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 2
    n_dim = 4
    batch_size = 8
    lr = 1e-4
    weight_decay = 1e-4
    n_epochs = 1000
    n_workers = 2

    transform = transforms.Compose(
        transforms=[
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = SampleDataset(path="./dataset/2_XYHW/train", transform=transform)
    val_dataset = SampleDataset(path="./dataset/2_XYHW/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # num_workers=n_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        # num_workers=n_workers
    )
    model = CNN18(3, n_samples * n_dim).to(device)  # FCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_loss_value = -1
    avg_loss = -1

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        val_epoch_loss = []

        progress_bar = tqdm(train_loader, leave=False)
        for (x, y) in progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            train_loss = loss.item()
            epoch_loss.append(train_loss)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=train_loss, mean_loss=avg_loss, val_loss=val_loss_value
            )

        with torch.no_grad():
            model.eval()
            val_progress_bar = tqdm(val_loader, leave=False)
            for (x, y) in val_progress_bar:
                val_progress_bar.set_description(f"Validation Epoch {epoch}")

                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = torch.nn.functional.mse_loss(y_pred, y)
                val_epoch_loss.append(loss.item())

        val_loss_value = np.mean(val_epoch_loss)
        avg_loss = np.mean(epoch_loss)

        # print(
        #     f"Epoch: {epoch}, Loss: {np.mean(epoch_loss):.6f}, Val_Loss: {np.mean(val_epoch_loss):.6f}"
        # )

        if not ((epoch + 1) % 5):
            torch.save(model.state_dict(), "./weights.pth")
