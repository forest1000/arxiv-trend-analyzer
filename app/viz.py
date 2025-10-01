from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd




def save_lineplot(series: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots()
    series.plot(ax=ax)
    ax.set_xlabel("week")
    ax.set_ylabel("papers")
    ax.set_title("arXiv weekly trend")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)