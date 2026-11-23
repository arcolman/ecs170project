"""Shared deterministic sampling and training-overlap checks for RNN/LSTM/GRU."""
import difflib
import re
import torch
from local_code.stage_4_code.Dataset_Loader_Generation import tokenize, detokenize


@torch.no_grad()
def generate(model, vocab, prompt='what did the', seed=1, temperature=.8,
             top_k=20, max_new_tokens=60):
    if temperature <= 0 or top_k < 1 or max_new_tokens < 1:
        raise ValueError('Sampling settings must be positive.')
    words = tokenize(prompt)
    if len(words) != 3 or not all(re.search('[a-z]', w) for w in words):
        raise ValueError('Supply exactly three starting words without punctuation.')
    lookup = {t: i for i, t in enumerate(vocab)}
    if any(w not in lookup for w in words):
        raise ValueError('Starting words must be in the training vocabulary.')
    device = next(model.parameters()).device
    # Sample on CPU so the random generator is independent of CUDA/MPS.
    rng = torch.Generator().manual_seed(seed)
    model.eval()
    ids = torch.tensor([[2] + [lookup[w] for w in words]], device=device)
    logits, hidden = model(ids)
    generated = list(words)
    stopped = False
    for _ in range(max_new_tokens):
        scores = logits[0, -1].float().cpu() / temperature
        scores[:3] = -float('inf')  # never emit padding, unknown or BOS
        values, indices = torch.topk(scores, min(top_k, len(vocab)-3))
        next_id = int(indices[torch.multinomial(values.softmax(0), 1, generator=rng)])
        if next_id == 3:
            stopped = True
            break
        generated.append(vocab[next_id])
        logits, hidden = model(torch.tensor([[next_id]], device=device), hidden)
    return {'prompt': prompt, 'seed': seed, 'temperature': temperature,
            'top_k': top_k, 'max_new_tokens': max_new_tokens,
            'stopped_on_eos': stopped, 'tokens': generated, 'text': detokenize(generated)}


def word_tokens(tokens):
    return [t for t in tokens if re.search('[a-z0-9]', t)]


def compare_to_training(sample, train_rows, n=5):
    words = word_tokens(sample['tokens'])
    # Exclude the supplied three-word prefix from copying/repetition measurements.
    continuation = words[3:]
    grams = [tuple(continuation[i:i+n]) for i in range(max(0, len(continuation)-n+1))]
    training_grams = set()
    best_size, best_row, best_start = 0, None, 0
    for row in train_rows:
        target = word_tokens(row['tokens'])
        training_grams.update(tuple(target[i:i+n]) for i in range(max(0, len(target)-n+1)))
        match = difflib.SequenceMatcher(None, continuation, target, autojunk=False).find_longest_match()
        if match.size > best_size:
            best_size, best_row, best_start = match.size, row, match.a
    return {'continuation_words': len(continuation), 'ngram_size': n,
            'ngram_positions': len(grams),
            'matching_training_ngram_positions': sum(g in training_grams for g in grams),
            'training_ngram_match_fraction': sum(g in training_grams for g in grams)/len(grams) if grams else None,
            'repeated_ngram_fraction': 1-len(set(grams))/len(grams) if grams else None,
            'longest_training_match_words': best_size,
            'longest_training_match': ' '.join(continuation[best_start:best_start+best_size]),
            'closest_training_id': best_row['id'] if best_row else None,
            'closest_training_text': best_row['text'] if best_row else None}
