"""The bridge to attacks: differentiate forecast loss with respect to the input."""
import argparse
import torch
from config import run_directory
from runtime import configure, load_run
from plotting import field_panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "era5"), default="synthetic")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = configure(42, args.device)
    output = run_directory(args.dataset, args.output_dir)
    model, datasets, stats, metadata = load_run(args.dataset, output, args.data, device)
    print(f"Source: {metadata['source']}")
    x, y = datasets["test"][0]                # [1,H,W] each.
    x = x.unsqueeze(0).to(device)             # [1,1,H,W]: add batch dimension.
    y = y.unsqueeze(0).to(device)
    x = x.clone().detach().requires_grad_(True)  # A fresh input that tracks gradients.
    prediction = model(x)
    loss = torch.nn.MSELoss()(prediction, y)
    loss.backward()                          # Model weights are frozen; input is not.
    gradient = x.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().any():
        raise RuntimeError("Expected a finite, nonzero input gradient.")
    print("x.grad.shape:", gradient.shape)
    print("x.grad.min():", gradient.min().item())
    print("x.grad.max():", gradient.max().item())
    print("x.grad.mean():", gradient.mean().item())
    # L is normalized MSE. Chain rule: dL/dT_K = dL/dx_normalized / std_K.
    physical_gradient = gradient / stats["std_k"]
    field_panels([gradient[0, 0].cpu().numpy(), physical_gradient[0, 0].cpu().numpy()],
                 ["Gradient wrt normalized input", "Gradient wrt Kelvin input"],
                 output / "input_gradient.png", differences=(0, 1), units="Loss sensitivity")


if __name__ == "__main__":
    main()
