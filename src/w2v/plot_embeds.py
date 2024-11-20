from pprint import pprint
from typing import Mapping

import numpy as np
import polars as pl
from sklearn.manifold import TSNE


def transform_tsne(embedding: np.ndarray):
    dim_reducer = TSNE(n_components=2, perplexity=5)
    return dim_reducer.fit_transform(embedding)


def plot_embeds(
    embedding_history: Mapping[int, np.ndarray], word_to_ix: Mapping[str, int]
) -> None:
    transformed = transform_tsne(embedding_history[0])
    pprint(transformed)
    df = pl.DataFrame(transformed)
    df = df.with_columns(
        pl.Series(
            name="token",
            values=[k for k in word_to_ix.keys()],
            dtype=pl.String,
        )
    )

    pprint(df)
