"""Train one configurable CNN on ERA5; validation chooses the saved weights."""
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import DATA, OUTPUT, VARIABLES
from data.era5 import load_data
from models.residual_cnn import ResidualCNN
from metrics import evaluate_loader
from plotting import training_plot
from runtime import configure, save_json, print_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['era5'], default='era5')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--variables', nargs='+', choices=VARIABLES, default=list(VARIABLES))
    parser.add_argument('--history', type=int, choices=[1, 2], default=2)
    parser.add_argument('--no-time-features', action='store_true')
    parser.add_argument('--direct-output', action='store_true', help='Disable correction and feature skip for comparison')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10, help='Stop after this many epochs without a lower validation loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width, args.patience) < 1 or args.lr <= 0 or args.depth < 3:
        parser.error('Positive settings and depth >= 3 required.')
    if len(set(args.variables)) != len(args.variables):
        parser.error('Do not repeat variables.')
    if args.output_dir.exists():
        parser.error('Results folder exists; choose a new --output-dir to preserve earlier runs.')
    device = configure(args.seed, args.device)
    datasets, stats, metadata, lat, lon = load_data(args.data, args.variables, args.history, not args.no_time_features)
    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir/'normalization.json', stats)
    save_json(args.output_dir/'data_metadata.json', metadata)
    save_json(args.output_dir/'training_config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    model = ResidualCNN(len(args.variables), args.history, not args.no_time_features,
                        args.width, args.depth, not args.direct_output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5, min_lr=1e-5)
    criterion = torch.nn.MSELoss()  # Mean across batch, FIVE normalized variables and cells.
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=args.batch_size)
    print(f"Counts: {metadata['counts']}; input {tuple(datasets['train'][0][0].shape)}; output {tuple(datasets['train'][0][1].shape)}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    history, best, stale = [], float('inf'), 0
    began = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        squared, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # [B,14,H,W] and [B,5,H,W] by default.
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()                   # Differentiate with respect to WEIGHTS.
            optimizer.step()                  # Update weights to reduce forecast error.
            squared += loss.item()*y.numel()
            count += y.numel()
        validation = evaluate_loader(model, validation_loader, stats, args.variables, metadata['units'], device)
        record = dict(epoch=epoch, train_mse=squared/count, validation=validation,
                      learning_rate=optimizer.param_groups[0]['lr'])
        history.append(record)
        score = validation['normalized_mse']
        summary = ' '.join(f"{v}={validation['variables'][v]['rmse']:.3f}{metadata['units'][i]}" for i, v in enumerate(args.variables))
        print(f'Epoch {epoch:02d}: train MSE={squared/count:.5f}; validation {summary}', flush=True)
        if score < best:
            best, stale = score, 0
            torch.save(dict(model_state=model.state_dict(), model_settings=model.settings,
                            stats=stats, metadata=metadata, epoch=epoch,
                            validation=validation, seed=args.seed), args.output_dir/'best_model.pt')
        else:
            stale += 1
        scheduler.step(score)                  # Only validation determines LR/checkpoint.
        save_json(args.output_dir/'history.json', history)
        if stale >= args.patience:
            print('Early stopping: validation has not improved.', flush=True)
            break
    training_plot(history, args.variables, metadata['units'], args.output_dir/'training_curves.png')
    checkpoint = torch.load(args.output_dir/'best_model.pt', map_location=device, weights_only=True)
    save_json(args.output_dir/'training_summary.json', dict(best_epoch=checkpoint['epoch'],
              epochs_run=len(history), seconds=time.monotonic()-began,
              parameters=sum(p.numel() for p in model.parameters()), validation=checkpoint['validation']))
    print(f"Saved best epoch {checkpoint['epoch']} to {args.output_dir}", flush=True)
    print_scores(checkpoint['validation'])
    # Test evaluation is a separate command; test weather never selects the epoch.


if __name__ == '__main__':
    main()
