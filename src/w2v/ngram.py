from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import ftfy
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
NUM_NEURONS = 128
NUM_EPOCH = 20

SEED = 42
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


def word_to_index(sentences):
    vocab = OrderedDict()
    ix = 0
    for w in sentences:
        if w not in vocab:
            vocab[w] = ix
            ix += 1
    return vocab


def train(ngrams, word_to_ix) -> NGramLanguageModeler:
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


def run_ngram() -> None:
    text = """
When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.
    """

    text = ftfy.fix_text(text)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_digit and not t.is_punct and not t.is_space
    ]
    ngrams = text_to_ngrams(tokens)

    word_to_ix = word_to_index(tokens)

    model_file = Path("./ngram.safetensor")
    if not model_file.exists():
        model = train(ngrams, word_to_ix)
        torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    run_ngram()
