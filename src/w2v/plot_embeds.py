from typing import Mapping

import altair as alt
import numpy as np
import polars as pl
from sklearn.manifold import TSNE


def transform_tsne(embedding: np.ndarray):
    dim_reducer = TSNE(n_components=2, perplexity=5)
    return dim_reducer.fit_transform(embedding)


def plot_embeds(
    embedding_history: Mapping[int, np.ndarray],
    word_to_ix: Mapping[str, int],
    highlights: list[str],
) -> None:
    print(highlights)
    embedding_dfs = []
    for nth in [0, 10, 20, 30, 40, 50]:
        transformed = transform_tsne(embedding_history[nth])
        df = pl.DataFrame(transformed)
        df = df.with_columns(
            pl.Series(
                name="token",
                values=[k for k in word_to_ix.keys()],
                dtype=pl.String,
            ),
            nth_epoch=pl.lit(nth),
        )
        df = df.with_columns(
            highlight=pl.col("token").is_in(highlights),
        )
        embedding_dfs.append(df)
    res_df = pl.concat(embedding_dfs)

    out_plot = "ngram.embeds.png"
    base = alt.Chart(res_df).encode(
        x="column_0:Q", y="column_1:Q", text="token:N", color="highlight:N"
    )
    plot = (
        alt.layer(
            base.mark_circle(),
            base.mark_text(align="left", baseline="middle", dx=5),
        )
        .properties(width=600, height=400)
        .facet(facet="nth_epoch:N", columns=3)
        .configure_axis(grid=False)
        .resolve_axis(
            x="independent",
            y="independent",
        )
        .resolve_scale(
            x="independent",
            y="independent",
        )
    )
    plot.save(
        out_plot,
        ppi=200,
    )
