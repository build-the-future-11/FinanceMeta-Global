#!/usr/bin/env python3
"""Build a reproducible cost figure and one-page evidence handout from retained JSON."""

import json
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

FONT_DIR = Path(
    os.environ.get("PRESENTATION_FONT_DIR", "/usr/share/fonts/truetype/dejavu")
)
pdfmetrics.registerFont(TTFont("HandoutBody", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("HandoutBold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("HandoutBody", normal="HandoutBody", bold="HandoutBold")
ROOT = Path(__file__).resolve().parent
summary = json.loads(
    (ROOT / "evidence/ledger-fault-enumeration-v1/summary.json").read_text()
)
(ROOT / "figures").mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
for name, label, color, marker in [
    ("strategy", "Declared strategy", "#a33434", "o"),
    ("buy_and_hold", "Buy-and-hold", "#156d88", "s"),
    ("cash", "Cash", "#666666", "^"),
]:
    rows = [x for x in summary["arithmetic_checks"] if x["method"] == name]
    ax.plot(
        [x["one_way_cost_bps"] for x in rows],
        [100 * float(x["net_total_return"]) for x in rows],
        label=label,
        color=color,
        marker=marker,
    )
ax.set(
    xlabel="Combined one-way cost (basis points)",
    ylabel="Hypothetical total return (%)",
    xticks=[0, 1, 5, 10, 25],
)
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=9)
fig.savefig(ROOT / "figures/cost_arithmetic.png", dpi=210)
plt.close(fig)

navy = colors.HexColor("#173646")
teal = colors.HexColor("#156d88")
styles = {
    "tag": ParagraphStyle(
        "tag", fontName="HandoutBold", fontSize=9, textColor=teal, spaceAfter=12
    ),
    "title": ParagraphStyle(
        "title",
        fontName="HandoutBold",
        fontSize=23,
        leading=27,
        textColor=navy,
        spaceAfter=12,
    ),
    "body": ParagraphStyle(
        "body", fontName="HandoutBody", fontSize=10.4, leading=14, spaceAfter=9
    ),
    "small": ParagraphStyle(
        "small", fontName="HandoutBody", fontSize=8.8, leading=11, spaceAfter=7
    ),
    "section": ParagraphStyle(
        "section",
        fontName="HandoutBold",
        fontSize=12,
        leading=15,
        textColor=navy,
        spaceBefore=8,
        spaceAfter=7,
    ),
}
story = [
    Paragraph("RAIOPS4FIN 2026 | AUTHOR-REVIEW PRESENTATION HANDOUT", styles["tag"]),
    Paragraph(
        "When a ledger fails,<br/>its performance stays hidden.", styles["title"]
    ),
    Paragraph(
        "Point-in-time controls for a frozen single-asset evaluation. All results below are software controls on manually specified synthetic values.",
        styles["body"],
    ),
]
table = Table(
    [
        ["58 controls", "55 invalid inputs", "15 cost comparisons"],
        [
            "All matched expectation",
            "All withheld every path",
            "Decimal error <= 3.44e-16",
        ],
    ],
    colWidths=[174] * 3,
)
table.setStyle(
    TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "HandoutBody"),
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "HandoutBold"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eef5f7")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]
    )
)
story += [table, Spacer(1, 10)]
story += [
    Paragraph("1. Check information before arithmetic", styles["section"]),
    Paragraph(
        "Each decision binds the latest feature and training-label availability. Inputs require timezone-aware times, strictly ordered decisions and contiguous future return intervals. A one-second future-information violation fails at every tested row.",
        styles["body"],
    ),
    Paragraph("2. Keep a pass separate from a gain", styles["section"]),
    Image(str(ROOT / "figures/cost_arithmetic.png"), width=360, height=205),
    Paragraph(
        "At 10 bps combined one-way cost, the valid strategy returns -2.446481%; buy-and-hold +1.158764%; cash 0%. These are hypothetical fixture calculations, not financial performance.",
        styles["small"],
    ),
    Paragraph("3. Preserve the review boundary", styles["section"]),
    Paragraph(
        "Hashes identify consumed bytes. They do not authenticate release times. The tool does not certify data licensing, simulate execution or finance, measure production adoption, or reopen a closed forecasting study.",
        styles["body"],
    ),
    Paragraph(
        "Talk sequence: contract (2 min) -> failed future-feature twin (2 min) -> fixed fault corpus and Decimal check (3 min) -> operational use and limits (3 min). Full source, reports and manuscript accompany this handout. Nothing has been submitted or accepted.",
        styles["small"],
    ),
]
SimpleDocTemplate(
    str(ROOT / "presentation_handout.pdf"),
    pagesize=(612, 792),
    rightMargin=45,
    leftMargin=45,
    topMargin=36,
    bottomMargin=32,
    title="Point-in-time ledger audit: presentation handout",
    author="",
).build(story)
