import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import ftfy
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plot_embeds import plot_embeds

CONTEXT_SIZE = 10
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


def train_loopy(ngrams, word_to_ix) -> NGramLanguageModeler:
    vocab_size = len(word_to_ix)
    print(f"vocab size={len(word_to_ix)}")

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
                [word_to_ix[w] for w in context], dtype=torch.long
            )
            # pprint(f"{context=} {target=}")
            # pprint(f"{context_idxs=} {word_to_ix[target]=}")

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)
            # pprint(log_probs)
            # pprint(f"{log_probs.shape=} {log_probs.ndim=}")
            # pprint(torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(
                log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long)
            )
            # pprint(f"{loss=}")
            # pprint(f"{log_probs[:, word_to_ix[target]]=}")

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
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_stack = nn.Sequential(
            nn.Linear(context_size * embedding_dim, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, vocab_size),
        )

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(
            (-1, CONTEXT_SIZE * EMBEDDING_DIM)
        )
        logits = self.ngram_stack(embeds)
        return F.log_softmax(logits, dim=1)


def fit(
    input_tensor, label_tensor, word_to_ix
) -> tuple[NGramModel, dict[int, np.ndarray]]:
    vocab_size = len(word_to_ix)
    print(f"vocab size={len(word_to_ix)}")

    losses = []
    loss_function = nn.NLLLoss(reduction="sum")
    model = NGramModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    pprint(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    embed_history = {}
    embed_history[0] = model.embeddings.weight.detach().numpy()
    for epoch in range(NUM_EPOCH):
        model.train()
        model.zero_grad()

        log_probs = model(input_tensor)
        loss = loss_function(log_probs, label_tensor)

        loss.backward()
        optimizer.step()
        pprint(f"{epoch=} {loss=}")
        if epoch % 10 == 0:
            embed_history[(epoch / 10) + 1] = (
                model.embeddings.weight.detach().clone().numpy()
            )
        losses.append(loss)

    embed_history[NUM_EPOCH] = model.embeddings.weight.detach().numpy()

    return model, embed_history


# TODO: plot distro of target work given context
# TODO: implement device


def generate_training_data(contexts, targets, word_to_ix):
    context_idxs = list(map(lambda c: [word_to_ix[w] for w in c], contexts))
    target_idxs = [word_to_ix[w] for w in targets]

    input_tensor = torch.tensor(context_idxs)
    label_tensor = torch.tensor(target_idxs)
    return input_tensor, label_tensor


def preprocess(text: str) -> str:
    return ftfy.fix_text(text)


def tokenize(text: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [t.text for t in doc if not t.is_digit and not t.is_space]


def word_to_index(sentences):
    vocab = OrderedDict()
    ix = 0
    for w in sentences:
        if w not in vocab:
            vocab[w] = ix
            ix += 1
    return vocab


def ngrams(tokens: list[str]):
    contexts = []
    targets = []
    for i in range(CONTEXT_SIZE, len(tokens)):
        contexts += [tokens[i - CONTEXT_SIZE : i]]
        targets += [tokens[i]]
    return contexts, targets


def generate_sentence(text_fspath: str, model, word_to_ix):
    test_text = ""
    with open(text_fspath, "r") as fIN:
        test_text = fIN.read()

    test_text = preprocess(test_text)
    test_tokens = tokenize(test_text)
    contexts, _ = ngrams(test_tokens)
    ix_to_word = {v: k for k, v in word_to_ix.items()}
    context_idxs = list(map(lambda c: [word_to_ix[w] for w in c], contexts))

    print("---------------inference------------------")
    sentences = contexts[0]
    with torch.inference_mode():
        cur_context = context_idxs[0]
        while len(sentences) <= len(test_tokens):
            test = torch.tensor(cur_context)
            pred_target_idx = model(test).softmax(dim=1).argmax()
            pred_target = ix_to_word[pred_target_idx.item()]
            sentences.append(pred_target)
            cur_context = cur_context[1:] + [pred_target_idx.item()]

        print(" ".join(sentences))
        print(test_text)


def run_ngram() -> None:
    train_text_fspath = sys.argv[1]
    test_text_fspath = sys.argv[2]

    train_text = ""
    with open(train_text_fspath, "r") as fIN:
        train_text = fIN.read()

    text = preprocess(train_text)
    tokens = tokenize(text)
    contexts, targets = ngrams(tokens)

    word_to_ix = word_to_index(tokens)
    input_tensor, label_tensor = generate_training_data(
        contexts, targets, word_to_ix
    )

    model_file = Path("./ngram.safetensor")
    embed_file = Path("./ngram.embed_history.pkl")
    if not model_file.exists():
        model, embed_history = fit(input_tensor, label_tensor, word_to_ix)
        torch.save(model.state_dict(), model_file)
        with open(embed_file, "wb") as f:
            pickle.dump(embed_history, f)

    print("Found previous model.")
    model = NGramLanguageModeler(len(word_to_ix), EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(torch.load(model_file, weights_only=True))

    generate_sentence(test_text_fspath, model, word_to_ix)

    with open(embed_file, "rb") as f:
        embed_history = pickle.load(f)
        plot_embeds(embed_history, word_to_ix)


if __name__ == "__main__":
    run_ngram()
