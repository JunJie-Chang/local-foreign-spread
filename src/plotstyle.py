"""seaborn / matplotlib 共用樣式 + 中文字型設定"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config as C

_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def setup():
    fm.fontManager.addfont(_CJK)
    name = fm.FontProperties(fname=_CJK).get_name()
    sns.set_theme(context=C.SNS_CONTEXT, style=C.SNS_STYLE, palette=C.SNS_PALETTE)
    plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = C.FIG_DPI
    return name


def save(fig, filename):
    C.FIG.mkdir(parents=True, exist_ok=True)
    path = C.FIG / filename
    fig.savefig(path, bbox_inches="tight", dpi=C.FIG_DPI)
    plt.close(fig)
    print("  fig ->", path)
