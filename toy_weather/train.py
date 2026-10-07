"""Train a tiny one-step forecaster. See README for the four learning experiments."""
import argparse
import torch
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, run_directory
from runtime import configure, get_data, save_json
from models.simple_cnn import SimpleCNN
from metrics import evaluate_loader
from plotting import training_plot, field_panels
from data.era5 import denormalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="synthetic")
    parser.add_argument("--data", help="Prepared ERA5 .npz file")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--size", type=int, default=16, help="Synthetic grid size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 2 or args.lr <= 0:
        parser.error("Use positive epochs, batch size and learning rate, and size >= 2.")
    device = configure(args.seed, args.device)
    datasets, stats, metadata = get_data(args.dataset, args.data, args.size, args.seed)
    output = run_directory(args.dataset, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Source: {metadata['source']}")
    save_json(output / "normalization.json", stats)
    save_json(output / "data_metadata.json", metadata)
    # Generic ML: shuffle training examples, but keep validation ordering stable.
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    padding = "circular" if args.dataset == "synthetic" else "replicate"
    model = SimpleCNN(padding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    history, best = [], float("inf")
    x, y = datasets["train"][0]
    x, y = [denormalize(t.numpy()[0], stats, celsius=True) for t in (x, y)]
    field_panels([x, y, y - x], ["Input", "True future", "Future − input"],
                 output / "example_pair.png", differences=(2,))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_squared, pixels = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)   # Both [B,1,H,W].
            optimizer.zero_grad()
            prediction = model(x)             # [B,1,H,W].
            loss = criterion(prediction, y)    # Scalar mean across batch and pixels.
            loss.backward()                   # Compute gradients of model weights.
            optimizer.step()                  # Update weights to reduce forecast loss.
            total_squared += loss.item() * y.numel()
            pixels += y.numel()
        metrics = evaluate_loader(model, validation_loader, device, stats["std_k"])
        row = {"epoch": epoch, "train_mse": total_squared / pixels,
               "validation_rmse_k": metrics["rmse_k"],
               "validation_rmse_normalized": metrics["rmse_normalized"]}
        history.append(row)
        print(f"Epoch {epoch:02d} | train MSE {row['train_mse']:.5f} | "
              f"validation RMSE {metrics['rmse_k']:.4f} K ({metrics['rmse_normalized']:.5f} normalized)")
        if metrics["rmse_k"] < best:
            best = metrics["rmse_k"]
            torch.save({"model_state": model.state_dict(), "dataset": args.dataset,
                        "padding_mode": padding, "stats": stats, "metadata": metadata,
                        "epoch": epoch, "validation_rmse_k": best}, output / "best_model.pt")
    save_json(output / "history.json", history)
    training_plot(history, output / "training_curves.png")
    print(f"Best validation RMSE: {best:.4f} K. Saved checkpoint and plots to {output}")


if __name__ == "__main__":
    main()
