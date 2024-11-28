import pickle
import sys
from functools import partial
from pathlib import Path
from pprint import pprint

import ftfy
import numpy as np
import polars as pl
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plot_embeds import plot_embeds
from plot_target_distro import plot_target_distro

CONTEXT_SIZE = 5
EMBEDDING_DIM = 16
NUM_NEURONS = 256
NUM_EPOCH = 500

SEED = 6969
torch.manual_seed(SEED)


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_stack = nn.Sequential(
            nn.Linear(context_size * embedding_dim, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, vocab_size),
        )

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        logits = self.ngram_stack(embeds)
        return F.log_softmax(logits, dim=1)


def text_to_ngrams(sentences: list[str]) -> list[tuple[list[str], str]]:
    return [
        (sentences[i - CONTEXT_SIZE : i], sentences[i])
        for i in range(CONTEXT_SIZE, len(sentences))
    ]


def train_loopy(ngrams, token_to_idx) -> NGramLanguageModeler:
    vocab_size = len(token_to_idx)
    print(f"vocab size={len(token_to_idx)}")

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    pprint(model)
    for name, param in model.named_parameters():
        pprint(f"Layer: {name} | Size: {param.size()}\n")
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCH):
        total_loss = 0
        for context, target in ngrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor(
                [token_to_idx[w] for w in context], dtype=torch.long
            )
            # pprint(f"{context=} {target=}")
            # pprint(f"{context_idxs=} {token_to_idx[target]=}")

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)
            # pprint(log_probs)
            # pprint(f"{log_probs.shape=} {log_probs.ndim=}")
            # pprint(torch.tensor([token_to_idx[target]], dtype=torch.long))

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(
                log_probs,
                torch.tensor([token_to_idx[target]], dtype=torch.long),
            )
            # pprint(f"{loss=}")
            # pprint(f"{log_probs[:, token_to_idx[target]]=}")

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        print(f"{epoch=} {total_loss=}")
        losses.append(total_loss)
    return model


class NGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramModel, self).__init__()
        self.embeddings = nn.Parameter(torch.rand(vocab_size, embedding_dim))
        self.ngram_stack = nn.Sequential(
            nn.Linear(context_size * embedding_dim, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, vocab_size),
        )

    def forward(self, inputs):
        embeds = self.embeddings[inputs].view(
            (-1, CONTEXT_SIZE * EMBEDDING_DIM)
        )
        logits = self.ngram_stack(embeds)
        return F.log_softmax(logits, dim=1)


def fit(
    input_tensor, label_tensor, token_to_idx
) -> tuple[NGramModel, dict[int, np.ndarray]]:
    vocab_size = len(token_to_idx)
    print(f"vocab size={len(token_to_idx)}")

    losses = []
    loss_function = nn.NLLLoss(reduction="sum")
    model = NGramModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    embed_history = {}
    embed_history[0] = model.embeddings.data.detach().clone().numpy()
    for epoch in range(NUM_EPOCH + 1):
        model.train()
        model.zero_grad()

        log_probs = model(input_tensor)
        loss = loss_function(log_probs, label_tensor)

        loss.backward()
        optimizer.step()
        pprint(f"{epoch=} {loss=}")
        if epoch % 10 == 0:
            nth = int(epoch / 10) + 1
            embed_history[nth] = model.embeddings.data.detach().clone().numpy()
        losses.append(loss)

    return model, embed_history


# TODO: implement to add randomness when generating text given context (like temperature)
# TODO: implement device


def generate_training_data(contexts, targets, token_to_idx):
    context_idxs = list(map(lambda c: [token_to_idx[w] for w in c], contexts))
    target_idxs = [token_to_idx[w] for w in targets]

    input_tensor = torch.tensor(context_idxs)
    label_tensor = torch.tensor(target_idxs)
    return input_tensor, label_tensor


def preprocess(text: str) -> str:
    return ftfy.fix_text(text)


def tokenize(text: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [t.text for t in doc if not t.is_digit and not t.is_space]


def ngrams(tokens: list[str]):
    contexts = []
    targets = []
    for i in range(CONTEXT_SIZE, len(tokens)):
        contexts += [tokens[i - CONTEXT_SIZE : i]]
        targets += [tokens[i]]
    return contexts, targets


def token_to_index(tokens: list[str]) -> dict[str, int]:
    word_to_idx = {}
    idx = 0
    for t in tokens:
        if t not in word_to_idx:
            word_to_idx[t] = idx
            idx += 1
    return word_to_idx


def nearest_neighbor(
    token: str,
    embedding: np.ndarray,
    token_to_idx: dict[str, int],
    top_k: int,
):
    idx = token_to_idx[token]
    token_embedding = torch.tensor(embedding[idx]).view(1, EMBEDDING_DIM)
    cos = (
        F.cosine_similarity(
            token_embedding, torch.tensor(embedding), dim=1, eps=1e-6
        )
        .detach()
        .numpy()
    )
    similarity = sorted(
        {float(v): i for i, v in enumerate(cos)}.items(), key=lambda x: x[0]
    )
    vocab_size = len(token_to_idx)
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    nearest_neighbor = [
        idx_to_token[i]
        for _, i in similarity[vocab_size - top_k - 1 : vocab_size]
    ]
    return nearest_neighbor


def generate_sentence(
    text_fspath: str,
    model: NGramModel,
    temperature: float,
    token_to_idx,
    top_k: int = 0,
    max_sentences: int = 8,
):
    test_text = ""
    with open(text_fspath, "r") as fIN:
        test_text = fIN.read()

    test_text = preprocess(test_text)
    test_tokens = tokenize(test_text)
    contexts, _ = ngrams(test_tokens)
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    context_idxs = map(lambda c: [token_to_idx[w] for w in c], contexts)

    sentences = contexts[0]
    with torch.inference_mode():
        given_context = next(context_idxs)
        dot_counter = 0
        while dot_counter < max_sentences:
            test = torch.tensor(given_context)
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
            given_context = given_context[1:] + [pred_token_idx.item()]
            if pred_token_idx == token_to_idx["."]:
                dot_counter += 1

        print(" ".join(sentences))
        print(test_text)


def get_target_distro(
    given_context: list[str],
    model: NGramModel,
    temperature: float,
    token_to_idx: dict[str, int],
):
    given_context_tensor = torch.tensor(
        [token_to_idx[t] for t in given_context]
    )
    with torch.inference_mode():
        logits = model(given_context_tensor) / temperature
        pred_targets = torch.softmax(logits, dim=1)
        # pred_targets = model(given_context_tensor).softmax(dim=1)
        df = pl.DataFrame(
            {
                "target_prob": pred_targets.squeeze().detach().numpy(),
                "pred_target": [t for t in token_to_idx.keys()],
            }
        )
        return df.with_columns(
            context=pl.lit(" ".join(given_context)),
        )


def run_ngram() -> None:
    train_text_fspath = sys.argv[1]
    test_text_fspath = sys.argv[2]

    train_text = ""
    with open(train_text_fspath, "r") as fIN:
        train_text = fIN.read()

    text = preprocess(train_text)
    tokens = tokenize(text)
    contexts, targets = ngrams(tokens)

    token_to_idx = token_to_index(tokens)
    input_tensor, label_tensor = generate_training_data(
        contexts, targets, token_to_idx
    )

    model_file = Path("./ngram.safetensor")
    embed_file = Path("./ngram.embed_history.pkl")
    if not model_file.exists():
        model, embed_history = fit(input_tensor, label_tensor, token_to_idx)
        torch.save(model.state_dict(), model_file)
        with open(embed_file, "wb") as f:
            pickle.dump(embed_history, f)

    print("Found previous model.")
    model = NGramModel(len(token_to_idx), EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(torch.load(model_file, weights_only=True))

    temperature = 2
    top_k = 5
    max_sentences = 10
    generate_sentence(
        test_text_fspath,
        model,
        temperature,
        token_to_idx,
        top_k=top_k,
        max_sentences=max_sentences,
    )

    # with open(test_text_fspath, "r") as fIN:
    #     test_text = fIN.read()
    #     test_text = preprocess(test_text)
    #     test_tokens = tokenize(test_text)
    #     contexts, _ = ngrams(test_tokens)
    #     distro_dfs = [
    #         df
    #         for df in map(
    #             partial(
    #                 get_target_distro,
    #                 model=model,
    #                 temperature=temperature,
    #                 token_to_idx=token_to_idx,
    #             ),
    #             contexts,
    #         )
    #     ]
    #     distro_df = pl.concat(distro_dfs)
    #     plot_target_distro(distro_df)

    # with open(embed_file, "rb") as f:
    #     embed_history = pickle.load(f)
    #     cluster = nearest_neighbor(
    #         "cat",
    #         embed_history[len(embed_history) - 1],
    #         token_to_idx,
    #         top_k=5,
    #     )
    #     plot_embeds(embed_history, token_to_idx, cluster)


if __name__ == "__main__":
    run_ngram()
