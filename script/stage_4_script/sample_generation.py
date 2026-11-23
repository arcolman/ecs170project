"""Generate from a saved checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import torch
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Generation_Helper import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--checkpoint', type=Path,
                        default=root/'result/stage_4_result/generation/gen_rnn_checkpoint.pt')
    parser.add_argument('--prompt', nargs=3, default=['what', 'did', 'the'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Method_RNN_Generation(**saved['config'])
    model.load_state_dict(saved['state_dict'])
    sample = generate(model, saved['vocab'], ' '.join(args.prompt), seed=args.seed,
                      temperature=args.temperature, top_k=args.top_k,
                      max_new_tokens=args.max_new_tokens)
    print(json.dumps(sample, indent=2))


if __name__ == '__main__':
    main()
