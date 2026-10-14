"""Evaluate saved weights once on the exploratory test period, with no updates."""
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from config import DATA, OUTPUT
from metrics import evaluate_loader
from plotting import forecast_plot
from runtime import configure, load_run, save_json, print_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['era5'], default='era5')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint-dir', type=Path, default=OUTPUT)
    parser.add_argument('--examples', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.examples < 0:
        parser.error('Use examples >= 0.')
    device = configure(device=args.device)
    model, datasets, stats, metadata, lat, lon = load_run(args.checkpoint_dir, args.data, device)
    scores = evaluate_loader(model, DataLoader(datasets['test'], batch_size=64), stats,
                             metadata['variables'], metadata['units'], device)
    scores['test_status'] = metadata['test_status']
    save_json(args.checkpoint_dir/'test_metrics.json', scores)
    print(metadata['test_status'])
    print_scores(scores)
    # Evenly spaced examples, not cases selected for looking unusually successful.
    import numpy as np
    import torch
    examples = []
    for number, index in enumerate(np.linspace(0, len(datasets['test'])-1, min(args.examples, len(datasets['test'])), dtype=int)):
        x, y = datasets['test'][int(index)]
        with torch.no_grad():
            prediction = model(x[None].to(device))[0].cpu().numpy()
            current = model.current_state(x[None].to(device))[0].cpu().numpy()
        forecast_plot(current, y.numpy(), prediction, stats, metadata['variables'],
                      metadata['units'], args.checkpoint_dir/f'test_example_{number}.png',
                      timestamp=str(datasets['test'].times[datasets['test'].triples[index, 1]]))
        examples.append(dict(sample=int(index), current_time=str(datasets['test'].times[datasets['test'].triples[index, 1]])))
    save_json(args.checkpoint_dir/'test_examples.json', examples)


if __name__ == '__main__':
    main()
