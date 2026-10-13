"""Differentiate forecast loss with respect to inputs, keeping weights frozen."""
import argparse
from pathlib import Path
import torch
from config import DATA, OUTPUT
from plotting import gradient_plot
from runtime import configure, load_run, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint-dir', type=Path, default=OUTPUT)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = configure(device=args.device)
    model, datasets, stats, metadata, lat, lon = load_run(args.checkpoint_dir, args.data, device)
    if not 0 <= args.sample < len(datasets['test']):
        parser.error('Sample is outside the test split.')
    x, y = datasets['test'][args.sample]
    x = x[None].to(device).clone().detach().requires_grad_(True)  # [1,14,H,W].
    y = y[None].to(device)                                    # [1,5,H,W].
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    loss.backward()                      # INPUT gradients; model weights are frozen.
    print('x.grad.shape:', x.grad.shape)
    print('x.grad.min():', x.grad.min().item())
    print('x.grad.max():', x.grad.max().item())
    print('x.grad.mean():', x.grad.mean().item())
    gradient = x.grad[0].cpu().numpy()
    finite, nonzero = bool(torch.isfinite(x.grad).all()), bool(x.grad.abs().max()>0)
    if not finite or not nonzero:
        raise RuntimeError('Expected finite, nonzero input gradients.')
    gradient_plot(gradient, metadata['variables'], metadata['history'], args.checkpoint_dir/'input_gradient.png')
    save_json(args.checkpoint_dir/'input_gradient.json', dict(shape=list(x.grad.shape),
              minimum=x.grad.min().item(), maximum=x.grad.max().item(), mean=x.grad.mean().item(),
              loss=loss.item(), finite=finite, nonzero=nonzero, sample=args.sample,
              units='derivative of normalized MSE with respect to normalized input'))


if __name__ == '__main__':
    main()
