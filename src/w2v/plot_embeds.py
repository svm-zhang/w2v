from pprint import pprint
from typing import Mapping

import altair as alt
import numpy as np
import polars as pl
from sklearn.manifold import TSNE


def transform_tsne(embedding: np.ndarray):
    dim_reducer = TSNE(n_components=2, perplexity=5)
    return dim_reducer.fit_transform(embedding)


def plot_embeds(
    embedding_history: Mapping[int, np.ndarray], word_to_ix: Mapping[str, int]
) -> None:
    for nth in [0, 24, 49]:
        transformed = transform_tsne(embedding_history[nth])
        df = pl.DataFrame(transformed)
        df = df.with_columns(
            pl.Series(
                name="token",
                values=[k for k in word_to_ix.keys()],
                dtype=pl.String,
            )
        )

        out_plot = f"ngram.embeds.{nth}.png"
        plot = (
            alt.Chart(df)
            .mark_circle(size=40)
            .encode(x="column_0:Q", y="column_1:Q")
            .properties(
                width=600,
                height=400,
            )
        ) + (
            alt.Chart(df)
            .mark_text(align="left", baseline="middle", dx=5)
            .encode(
                x="column_0:Q",
                y="column_1:Q",
                text="token:N",
            )
        )
        plot.save(
            out_plot,
            ppi=200,
        )
