from functools import partial

import hydra
import matplotlib.ticker as ticker
import pandas as pd
from fanda.utils import close_fig, save_fig
from fanda.visualizations import (
    add_legend,
    annotate_axis,
    decorate_axis,
    lineplot,
    pointplot,
)
from fanda.wandb_client import fetch_wandb
from omegaconf import DictConfig, OmegaConf
from scipy.stats import trim_mean
from tqdm import tqdm


def plot_interval_estimates(df, key, group_by, path):
    (
        pointplot(
            df=df,
            x=key,
            y=group_by,
            hue=group_by,
            palette="colorblind",
            order=df.groupby(group_by)[key].mean().sort_values().index.tolist(),
            capsize=0.2,
            dodge=True,
            estimator=partial(trim_mean, proportiontocut=0.25),
        )
        .pipe(
            annotate_axis,
            xlabel="Mean Returns",
            title="IQM",
            grid_alpha=0.25,
        )
        .pipe(decorate_axis, ticklabelsize="xx-large", wrect=5, spines=["bottom"])
        .pipe(save_fig, name=path)
        .pipe(close_fig)
    )


def plot_sample_efficiency(df, key, group_by, path):
    def set_major_formatter(fanda):
        formatter = ticker.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}")
        fanda.ax.xaxis.set_major_formatter(formatter)
        fanda.ax.xaxis.set_minor_formatter(formatter)
        return fanda

    labels = df[group_by].unique()
    (
        lineplot(
            df,
            x="_step",
            y=key,
            hue=group_by,
            palette="colorblind",
            estimator=partial(trim_mean, proportiontocut=0.25),
            err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis,
            xlabel="Number of Frames (in millions)",
            ylabel="IQM Mean Returns",
            labelsize="xx-large",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(set_major_formatter)
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=path)
        .pipe(close_fig)
    )


def plot_performance_profile(df, key, group_by, path):
    df["probability"] = 1.0 - df.groupby(group_by)[key].rank(method="max", pct=True)

    anchors = df[[group_by]].drop_duplicates().copy()
    anchors[key] = 0.0
    anchors["probability"] = 1.0

    df = pd.concat([anchors, df], ignore_index=True).sort_values(
        by=[group_by, key], ascending=False
    )
    (
        lineplot(
            df,
            x=key,
            y="probability",
            hue=group_by,
            palette="colorblind",
            drawstyle="steps-post",
        )
        .pipe(
            annotate_axis,
            xlabel=r"Score $(\tau)$",
            ylabel=r"Fraction of runs with score $\geq \tau$",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(add_legend, labels=df[group_by].unique())
        .pipe(save_fig, name=path)
        .pipe(close_fig)
    )


@hydra.main(version_base=None, config_path="../config/plot", config_name="minatar")
def main(cfg: DictConfig):
    key = f"{cfg.prefix}/episode_returns"
    group_by = cfg.group_by
    filters = {
        "config.environment.env_id": {"$regex": cfg.environment},
        "state": {"$in": list(cfg.states)} if cfg.states else None,
        "config.algorithm.name": (
            {"$in": list(cfg.algorithms)} if cfg.algorithms else None
        ),
        "config.torso.name": {"$in": list(cfg.torsos)} if cfg.torsos else None,
    }

    df = fetch_wandb(
        cfg.entity,
        cfg.project,
        keys=[key, "_step"],
        filters=filters,
    )

    plot_dir = f"plots/{cfg.environment}/{group_by}"

    print("Plotting Interval Estimates...")
    plot_interval_estimates(
        df.groupby([group_by, "environment.env_id", "run_id"], as_index=False)[
            key
        ].max(),
        key,
        group_by,
        f"{plot_dir}/interval_estimates",
    )

    print("Plotting Performance Profile...")
    plot_performance_profile(
        df.groupby([group_by, "environment.env_id", "run_id"], as_index=False)[
            key
        ].max(),
        key,
        group_by,
        f"{plot_dir}/performance_profile",
    )

    print("Plotting Sample Efficiency...")
    plot_sample_efficiency(
        df,
        key,
        group_by,
        f"{plot_dir}/sample_efficiency",
    )

    env_ids = df["environment.env_id"].unique()
    for env_id in tqdm(env_ids, leave=False):
        print(f"Plotting Sample Efficiency for {env_id}...")
        plot_sample_efficiency(
            df[df["environment.env_id"] == env_id],
            key,
            group_by,
            f"{plot_dir}/{env_id}",
        )


if __name__ == "__main__":
    main()
