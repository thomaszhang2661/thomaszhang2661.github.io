"""Grouped bar chart (SVG, no dependencies) for the FP8 KV post. python3 _tools/chart.py -> assets/fp8_qwen2_{en,zh}.svg"""
import html

DATA = [  # model, BF16, FP8, FP8 + layer 0 BF16 (None: not run)
    ("R1-Distill-Qwen-7B", 71.0, 0.5, 71.5),
    ("Qwen2.5-7B-Instruct", 73.5, 1.5, 72.0),
    ("R1-Distill-Qwen-1.5B", 61.0, 0.0, 61.0),
    ("R1-Distill-Qwen-14B", 70.0, 73.0, None),
    ("R1-Distill-Llama-8B", 66.6, 65.6, 66.6),
]
TEXT = {
    "en": dict(title="MATH-500 accuracy in vLLM: FP8 KV cache vs BF16",
               sub="vLLM 0.33.0, H200, greedy, 4,096 tokens, default FP8 scales",
               series=["BF16 KV", "FP8 KV", "FP8, layer 0 in BF16"], notrun="n/a",
               base=["Qwen2 base", "Llama base"],
               foot="First 200 problems; R1-Distill-Llama-8B: all 500. Full numbers in the table below."),
    "zh": dict(title="vLLM 中 MATH-500 正确率：FP8 KV cache 对 BF16",
               sub="vLLM 0.33.0，H200，贪心解码，最多 4096 token，默认 FP8 缩放因子",
               series=["BF16 KV", "FP8 KV", "FP8，第 0 层用 BF16"], notrun="未测",
               base=["Qwen2 底座", "Llama 底座"],
               foot="前 200 题；R1-Distill-Llama-8B 为全部 500 题。完整数字见下表。"),
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz reference palette, slots 1-3 (validated, light)
SURF, T1, T2, T3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e4e3df"
W, H = 740, 460
L, R, TOP, BOT = 64, 24, 118, 92
PW, PH = W - L - R, H - TOP - BOT
YMAX = 80
BW, GAP = 26, 2
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Noto Sans SC', Helvetica, Arial, sans-serif"


def y(v):
    return TOP + PH * (1 - v / YMAX)


def bar(x, v, color, tip):
    """bar with 4px rounded top corners, square at the baseline"""
    y0, y1 = y(0), y(v)
    h = y0 - y1
    r = min(4, h / 2, BW / 2)
    if h < 0.5:  # zero: a 1px tick on the baseline so the slot is visible
        return f'<rect x="{x:.1f}" y="{y0 - 1:.1f}" width="{BW}" height="1" fill="{color}"><title>{tip}</title></rect>'
    d = (f"M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + BW - r:.1f} "
         f"Q{x + BW:.1f},{y1:.1f} {x + BW:.1f},{y1 + r:.1f} V{y0:.1f} Z")
    return f'<path d="{d}" fill="{color}"><title>{tip}</title></path>'


def svg(lang):
    t = TEXT[lang]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="{html.escape(t["title"])}" style="font-family:{FONT};max-width:{W}px">',
           f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
           f'<text x="{L}" y="34" font-size="20" font-weight="600" fill="{T1}">{html.escape(t["title"])}</text>',
           f'<text x="{L}" y="58" font-size="13" fill="{T2}">{html.escape(t["sub"])}</text>']
    # legend (always present: 3 series)
    lx = L
    for c, s in zip(COLORS, t["series"]):
        out.append(f'<rect x="{lx}" y="76" width="12" height="12" rx="3" fill="{c}"/>')
        out.append(f'<text x="{lx + 18}" y="86.5" font-size="13" fill="{T1}">{html.escape(s)}</text>')
        lx += 18 + 8.2 * len(s) + (26 if lang == "en" else 60)
    # grid + y axis labels (recessive)
    for v in range(0, YMAX + 1, 20):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{L - 10}" y="{y(v) + 4:.1f}" font-size="12" fill="{T3}" text-anchor="end">{v}%</text>')
    gw = PW / len(DATA)
    for i, (m, *vals) in enumerate(DATA):
        gx = L + gw * i + (gw - (3 * BW + 2 * GAP)) / 2
        for j, (v, c) in enumerate(zip(vals, COLORS)):
            x = gx + j * (BW + GAP)
            if v is None:
                out.append(f'<text x="{x + BW / 2:.1f}" y="{y(0) - 6:.1f}" font-size="11" fill="{T3}" '
                           f'text-anchor="middle">{t["notrun"]}</text>')
                continue
            out.append(bar(x, v, c, f"{m} · {t['series'][j]}: {v:.1f}%"))
            if j == 1:  # selective direct labels: the FP8 bars (the collapse) only
                ly = min(y(v) - 7, y(0) - 7)
                out.append(f'<text x="{x + BW / 2:.1f}" y="{ly:.1f}" font-size="10.5" font-weight="600" fill="{T1}" '
                           f'text-anchor="middle">{v:.1f}%</text>')
        cx = L + gw * i + gw / 2
        out.append(f'<text x="{cx:.1f}" y="{y(0) + 20:.1f}" font-size="11.5" fill="{T1}" text-anchor="middle">'
                   f'{html.escape(m)}</text>')
    # base-model brackets under the labels
    for (a, b), name in (((0, 3), t["base"][0]), ((4, 4), t["base"][1])):
        x1, x2 = L + gw * a + 10, L + gw * (b + 1) - 10
        yb = y(0) + 34
        out.append(f'<path d="M{x1:.1f},{yb:.1f} v5 H{x2:.1f} v-5" fill="none" stroke="{T3}" stroke-width="1"/>')
        out.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{yb + 20:.1f}" font-size="12" fill="{T2}" '
                   f'text-anchor="middle">{html.escape(name)}</text>')
    out.append(f'<text x="{L}" y="{H - 10}" font-size="11.5" fill="{T3}">{html.escape(t["foot"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for lang in TEXT:
        os.makedirs(os.path.join(here, "_includes"), exist_ok=True)
        for d in ("assets", "_includes"):  # _includes: inlined into the post (hover titles work); assets: direct link
            open(os.path.join(here, d, f"fp8_qwen2_{lang}.svg"), "w").write(svg(lang))
        print("wrote", lang)
