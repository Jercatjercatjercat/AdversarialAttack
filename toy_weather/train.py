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
    parser.add_argument("--dataset", choices=("synthetic", "era5"),
                        default="synthetic")    # choose dataset to train on
    # path to the prepared real-data file
    parser.add_argument("--data", help="Prepared ERA5 .npz file")
    # where to save the model and plots
    parser.add_argument(
        "--output-dir", help="Where to save the model and plots")
    # One epoch visits all training examples: 4,000 pairs at batch size 64 give
    # 63 weight updates. Shuffle the examples again for the next epoch.
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    # the number of samples per batch (in a sum notation)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--size", type=int, default=16,
                        help="Synthetic grid size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args()  # read and interpret the command line arguments
    if args.epochs < 1 or args.batch_size < 1 or args.size < 2 or args.lr <= 0:
        parser.error(
            "Use positive epochs, batch size and learning rate, and size >= 2.")
    device = configure(args.seed, args.device)
    # generate or load the dataset based on command line arguments
    datasets, stats, metadata = get_data(
        args.dataset, args.data, args.size, args.seed)
    # Example contents below use the default synthetic setup (size=16, seed=42).
    # datasets is a dictionary of PyTorch datasets, not one big tensor:
    # {"train": 4,000 pairs, "validation": 500 pairs, "test": 500 pairs}.
    # x, y = datasets["train"][0] gets the first input and its true future.
    # Each is a normalized tensor shaped [1,16,16]; there is no batch axis yet.
    # Example x[0,:3,:3] (top-left corner of that first temperature map):
    # [[ 0.1270, -0.0154,  0.1838],
    #  [ 0.4303,  0.1892,  0.3649],
    #  [ 0.7022,  0.4600,  0.5540]]. These values are NOT Kelvin.
    # stats is a dictionary calculated from training inputs only, approximately:
    # {"mean_k": 285.014, "std_k": 4.478}.
    # normalized = (temperature_K - mean_k) / std_k; 0 means about 285.014 K.
    # metadata describes where the data came from:
    # {"source": "synthetic", "size": 16, "seed": 42}.
    # Real-file data has different counts, statistics and source information.

    # Choose a results folder: outputs/synthetic by default for synthetic data,
    # or the folder you supplied with --output-dir. No tensors change here.
    output = run_directory(args.dataset, args.output_dir)
    # Create it and any missing parent folders; an existing folder is fine.
    output.mkdir(parents=True, exist_ok=True)

    # Show the data source in the terminal, for example: Source: synthetic.
    print(f"Source: {metadata['source']}")

    # Save small, readable JSON files, not the temperature maps themselves.
    # Here '/' joins a folder and filename; it is not numerical division.
    # Keep the training mean/std so normalized values can be converted to Kelvin.
    save_json(output / "normalization.json", stats)
    # Record the source and settings, e.g. synthetic, size=16, seed=42.
    save_json(output / "data_metadata.json", metadata)

    # Generic ML: shuffle training examples, but keep validation ordering stable.
    # DataLoader groups individual [1,16,16] maps into [B,1,16,16] batches.
    # With batch size 64, B=64 except for the last training batch, where B=32.
    # Split the 4000 training pairs into 63 batches of 64 and 1 batch of 32.
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True)
    # Split the 500 validation pairs into 7 batches of 64 and 1 batch of 52.
    validation_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size)
    # how to deal with the edges of the convolutional kernel:
    # circular: wrap around the edges (synthetic data)
    # replicate: repeat the edge values (ERA5 data)
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
            # Scalar mean across batch and pixels.
            loss = criterion(prediction, y)
            # Compute gradients of model weights.
            loss.backward()
            # Update weights to reduce forecast loss.
            optimizer.step()
            total_squared += loss.item() * y.numel()
            pixels += y.numel()
        metrics = evaluate_loader(
            model, validation_loader, device, stats["std_k"])
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
    print(
        f"Best validation RMSE: {best:.4f} K. Saved checkpoint and plots to {output}")


if __name__ == "__main__":
    main()
