#!/usr/bin/env python3
"""Render a 1200x630 Open Graph preview card for a Lumo FlyWheel volume.

Social crawlers (LinkedIn, Slack, X) will not render a large preview card
without an og:image, so every volume needs one. The card mirrors the site's
own palette and type so a shared link looks like the page it points at.

The generated PNG is fetched by the crawler, never by the page itself, so the
volumes stay zero-external-request documents.

Usage:
    python3 tools/make_og_card.py            # regenerate every card
    python3 tools/make_og_card.py vol10      # just one
"""
import sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = "#111111"
INK = "#f2efe7"
TEXT = "#d8d3c8"
MUTED = "#a8a196"
RULE = "#3b3934"
CYAN = "#8bd3dd"
GREEN = "#a7d98d"
AMBER = "#e7c76f"

SERIF = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
SERIF_R = "/System/Library/Fonts/Supplemental/Georgia.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"

# slug -> (kicker, title, stat, stat_label, accent)
CARDS = {
    "vol10": ("LUMO FLYWHEEL / VOL. X", "Twenty-Seven Milliseconds, Shipped",
              "−27 ms", "per step, four concurrent requests — byte-identical, shipped", GREEN),
    "vol9": ("LUMO FLYWHEEL / VOL. IX", "Where Every Millisecond Went",
             "232.8 ms", "the decode step, against a 119.7 ms hardware floor", CYAN),
    "vol8": ("LUMO FLYWHEEL / VOL. VIII", "Every Lever We Pulled",
             "8 / 16", "speed attempts that paid, out of sixteen tried", CYAN),
    "vol7": ("LUMO FLYWHEEL / VOL. VII", "Keep or Replay",
             "78%", "of the throughput gap traced to the committer", AMBER),
    "vol6": ("LUMO FLYWHEEL / VOL. VI", "One Bit in the Last Place",
             "1 ULP", "of reassociation, and a superset that stopped holding", AMBER),
    "vol5": ("LUMO FLYWHEEL / VOL. V", "The Stateless Tree",
             "0", "give-ups, after the attention-state garble fix", GREEN),
    "vol4": ("LUMO FLYWHEEL / VOL. IV", "Bit-Exact Prefix Caching",
             "EXACT", "prefix reuse under speculative decoding", CYAN),
    "vol3": ("LUMO FLYWHEEL / VOL. III", "Lossless GDN Tree Scan",
             "LOSSLESS", "a branched verifier on a recurrent hybrid", CYAN),
    "vol2": ("LUMO FLYWHEEL / VOL. II", "The Round-F Reversal",
             "REVERSED", "what a fuller ablation did to the verdict", AMBER),
    "vol1": ("LUMO FLYWHEEL / VOL. I", "Track-B Round-4b Ablation",
             "5x", "the real-workload audit that started the series", CYAN),
}


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(slug, kicker, title, stat, stat_label, accent, out):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # faint grid, same 56px rhythm as the site background
    for x in range(0, W, 56):
        d.line([(x, 0), (x, H)], fill="#161616", width=1)
    for y in range(0, H, 56):
        d.line([(0, y), (W, y)], fill="#151515", width=1)

    pad = 72
    f_kick = font(MONO, 21)
    f_stat = font(SERIF, 92)
    f_label = font(MONO, 20)
    f_by = font(MONO, 19)

    d.text((pad, 62), kicker, font=f_kick, fill=CYAN)

    # title, shrink to fit two lines
    size = 76
    while size > 44:
        f_title = font(SERIF, size)
        lines = wrap(d, title, f_title, W - 2 * pad)
        if len(lines) <= 2:
            break
        size -= 4
    y = 122
    for ln in lines:
        d.text((pad, y), ln, font=f_title, fill=INK)
        y += int(size * 1.16)

    d.line([(pad, 392), (W - pad, 392)], fill=RULE, width=1)

    d.text((pad, 424), stat, font=f_stat, fill=accent)
    sw = d.textlength(stat, font=f_stat)
    for i, ln in enumerate(wrap(d, stat_label, f_label, W - pad * 2 - sw - 34)):
        d.text((pad + sw + 34, 452 + i * 28), ln, font=f_label, fill=MUTED)

    d.text((pad, H - 66), "Mark Ma  ·  ML systems / inference  ·  Qwen 3.6-27B FP8 / GB10",
           font=f_by, fill=TEXT)
    d.rectangle([0, H - 8, W, H], fill=accent)

    img.save(out, "PNG", optimize=True)
    print(f"{out}  {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    want = sys.argv[1:] or list(CARDS)
    for slug in want:
        k, t, s, sl, a = CARDS[slug]
        render(slug, k, t, s, sl, a, f"og-{slug}.png")
