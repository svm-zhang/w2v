import altair as alt
import polars as pl


def plot_target_distro(df: pl.DataFrame):
    print(df.head())

    out_plot = "ngram.target.dist.png"
    base = alt.Chart(df).encode(x="pred_target:N", y="target_prob")
    plot = (
        alt.layer(base.mark_bar(cornerRadius=5))
        .properties(width=1000, height=800)
        .facet(facet="context:N", columns=10)
        .configure_axis(grid=False)
        .resolve_axis(x="independent")
        .resolve_scale(x="independent")
    )

    plot.save(out_plot, ppi=200)
