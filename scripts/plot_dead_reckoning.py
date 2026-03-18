from functools import partial

import matplotlib.ticker as ticker
from fanda import transforms
from fanda.utils import close_fig, save_fig
from fanda.visualizations import add_legend, annotate_axis, decorate_axis, lineplot
from fanda.wandb_client import fetch_wandb
from scipy.stats import trim_mean

MODES = ["training", "evaluation"]
BOUNDS = (0.0, 64.0)


def get_network(df):
    """Extract human-readable network name from config fields."""

    def func(row):
        torso = str(row.get("torso._target_", ""))
        cell = str(row.get("torso.cell._target_", ""))

        if "FFN" in torso:
            return "MLP"
        if "RNN" in torso:
            return "GRU"
        if "Memoroid" in torso:
            if "SALTCell" in cell:
                return "SALT"
            if "GatedDeltaRule" in cell:
                return "Gated DeltaNet"
            if "DeltaRule" in cell:
                return "DeltaNet"
            if "LinearAttention" in cell:
                return "Linear Transformer"
        return torso.split(".")[-1]

    df["network"] = df.apply(func, axis=1)
    return df


def set_major_formatter(fanda):
    formatter = ticker.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}")
    fanda.ax.xaxis.set_major_formatter(formatter)
    return fanda


def plot_mode(df, mode):
    metric = f"{mode}/mean_episode_returns"
    df = transforms.normalize(df, column=metric, min=BOUNDS[0], max=BOUNDS[1])
    df = df[df["_step"] <= 50_000_000]
    labels = sorted(df["network"].unique())

    (
        lineplot(
            df,
            x="_step",
            y=metric,
            hue="network",
            hue_order=labels,
            palette="colorblind",
            estimator=partial(trim_mean, proportiontocut=0.25),
            err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis,
            xlabel="Steps (in millions)",
            ylabel="Normalized IQM Episodic Returns",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(set_major_formatter)
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=f"plots/dead_reckoning/{mode}")
        .pipe(close_fig)
    )


def main():
    df = fetch_wandb(
        "noahfarr",
        "salt",
        filters={
            "config.environment.env_id": "DeadReckoning-v1",
            "config.algorithm.name": "pqn",
        },
    ).pipe(get_network)

    for mode in MODES:
        plot_mode(df, mode)


if __name__ == "__main__":
    main()
