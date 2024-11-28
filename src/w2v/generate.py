import torch

from .model import NGramModel


def generate_sentence(
    init_context: list[str],
    model: NGramModel,
    temperature: float,
    token_to_idx,
    top_k: int = 0,
    max_sentences: int = 8,
) -> str:
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    given_context_idx = [token_to_idx(t) for t in init_context]

    sentences = init_context
    with torch.inference_mode():
        dot_counter = 0
        while dot_counter < max_sentences:
            test = torch.tensor(given_context_idx)
            logits = model(test) / temperature
            if top_k > 0:
                # idx_to_remove = (
                #     logits
                #     < torch.topk(logits, k=top_k, dim=1)[0][..., -1, None]
                # )
                # logits[idx_to_remove] = -float("Inf")
                _, top_k_idx = torch.topk(logits, k=top_k, dim=1)
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, top_k_idx, True)
                logits[~mask] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            pred_token_idx = torch.multinomial(
                probs,
                num_samples=1,
            )
            pred_token = idx_to_token[pred_token_idx.item()]
            sentences.append(pred_token)
            given_context_idx = given_context_idx[1:] + [pred_token_idx.item()]
            if pred_token_idx == token_to_idx["."]:
                dot_counter += 1

        return " ".join(sentences)
