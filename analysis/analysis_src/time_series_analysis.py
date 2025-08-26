import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensor_over_cycles(df: pd.DataFrame, sensor: str, n_engines: int = 4) -> None:
    """
    Plot a given sensor versus cycle for a few random engines.
    df must have columns: ['engine_id', 'cycle', sensor]
    """
    # pick a few engines to avoid overplotting
    engines = df["engine_id"].drop_duplicates().sample(min(n_engines, df["engine_id"].nunique()), random_state=42)
    plt.figure(figsize=(10, 6))
    for eid in engines:
        sub = df[df["engine_id"] == eid]
        plt.plot(sub["cycle"], sub[sensor], marker="", linewidth=1, alpha=0.9, label=f"engine {eid}")
    plt.title(f"{sensor} over cycles (sample of {len(engines)} engines)")
    plt.xlabel("cycle"); plt.ylabel(sensor)
    plt.legend()
    plt.show()

def plot_cycles_per_engine(df: pd.DataFrame) -> None:
    """
    Show distribution of number of cycles each engine ran in TRAIN set.
    """
    counts = df.groupby("engine_id")["cycle"].max().reset_index(name="max_cycle")
    plt.figure(figsize=(8, 5))
    sns.histplot(counts["max_cycle"], bins=30, kde=True)
    plt.title("Distribution of max cycles per engine (train)")
    plt.xlabel("max cycle"); plt.ylabel("count of engines")
    plt.show()

def plot_settings_relationship(df: pd.DataFrame, sensor: str) -> None:
    """
    Visualize how operational settings relate to a sensor (scatter in 3 small panels).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, ax in enumerate(axes, start=1):
        sns.scatterplot(x=df[f"setting_{i}"], y=df[sensor], s=8, alpha=0.4, ax=ax)
        ax.set_title(f"{sensor} vs setting_{i}")
        ax.set_xlabel(f"setting_{i}"); ax.set_ylabel(sensor if i == 1 else "")
    fig.suptitle(f"{sensor} vs operational settings")
    plt.tight_layout()
    plt.show()
