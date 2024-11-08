import re
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

SEED = 1
torch.manual_seed(SEED)


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_stack = nn.Sequential(
            nn.Linear(context_size * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        logits = self.ngram_stack(embeds)
        return F.log_softmax(logits, dim=1)


def text_to_ngrams(sentences: list[str]) -> list[tuple[list[str], str]]:
    return [
        (
            [sentences[i - j - 1] for j in range(CONTEXT_SIZE)],
            sentences[i],
        )
        for i in range(CONTEXT_SIZE, len(sentences))
    ]


def word_to_index(sentences):
    vocab = {}
    ix = 0
    for w in sentences:
        if w not in vocab:
            vocab[w] = ix
            ix += 1
    return vocab


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

    text = re.sub(r"[^\w\s]", "", text)
    sentences = text.split()
    ngrams = text_to_ngrams(sentences)

    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.
    # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
    # Print the first 3, just so you can see what they look like.

    word_to_ix = word_to_index(sentences)
    vocab_size = len(word_to_ix)

    print(f"vocab size={len(word_to_ix)}")

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
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

    # To get the embedding of a particular word, e.g. "beauty"
    print(model.embeddings.weight[word_to_ix["beauty"]])
    # tensor([ 0.2509,  0.3671, -0.5116, -0.0997, -0.5093,  1.2391,  0.4230,  0.2663,
    #          0.0769, -1.0799], grad_fn=<SelectBackward0>)

    cos = nn.CosineSimilarity(dim=0)
    cos_similarities = []
    for word in word_to_ix.keys():
        cos_similarities.append(
            (
                cos(
                    model.embeddings.weight[word_to_ix["beauty"]],
                    model.embeddings.weight[word_to_ix[word]],
                ).item(),
                word,
            )
        )
    print(sorted(cos_similarities)[:5])


if __name__ == "__main__":
    run_ngram()
