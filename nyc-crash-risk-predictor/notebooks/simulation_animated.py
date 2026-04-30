"""
NYC Crash Risk Predictor — Monte Carlo Simulation Core
Video 3: Building the Simulation  |  Animated Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import sys

# ── CONFIGURATION ────────────────────────────────────────────────────────────

ZIP_CODE     = "10003"
HOUR         = 17
DAY_OF_WEEK  = "Tuesday"
N_TRIALS     = 10_000
RANDOM_SEED  = 42

OBSERVED_CRASHES = 42
OBSERVED_HOURS   = 52

WEATHER_PROBS = {
    "clear": 0.65,
    "rain":  0.28,
    "snow":  0.07,
}
WEATHER_MULTIPLIERS = {
    "clear": 1.00,
    "rain":  1.35,
    "snow":  1.72,
}

alpha_post = 1.0 + OBSERVED_CRASHES   # 43
beta_post  = 1.0 + OBSERVED_HOURS     # 53


# ── RUN FULL SIMULATION ───────────────────────────────────────────────────────

print("Running simulation …")
rng = np.random.default_rng(RANDOM_SEED)
results = np.empty(N_TRIALS, dtype=int)
weather_labels = list(WEATHER_PROBS.keys())
weather_probs  = list(WEATHER_PROBS.values())

for i in range(N_TRIALS):
    lam        = rng.gamma(shape=alpha_post, scale=1 / beta_post)
    weather    = rng.choice(weather_labels, p=weather_probs)
    lam_adj    = lam * WEATHER_MULTIPLIERS[weather]
    results[i] = rng.poisson(lam_adj)

p_crash      = (results >= 1).mean()
mean_crashes = results.mean()
ci_low       = np.percentile(results, 2.5)
ci_high      = np.percentile(results, 97.5)
running_p    = np.cumsum(results >= 1) / np.arange(1, N_TRIALS + 1)

# Running 95 % CI half-width (binomial SE)
running_n  = np.arange(1, N_TRIALS + 1)
running_se = np.sqrt(running_p * (1 - running_p) / running_n)

print(f"P(≥1 crash)  = {p_crash:.1%}")
print(f"Mean crashes = {mean_crashes:.3f}")
print(f"95 % CI      = [{ci_low:.0f}, {ci_high:.0f}]")


# ── COLOUR PALETTE ────────────────────────────────────────────────────────────

BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#58a6ff"
YELLOW = "#e3b341"
PURPLE = "#d2a8ff"
ORANGE = "#ffa657"

GLOW = [pe.withStroke(linewidth=4, foreground=BG)]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,
    "text.color":       WHITE,
    "axes.labelcolor":  WHITE,
    "xtick.color":      GRAY,
    "ytick.color":      GRAY,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "font.family":      "DejaVu Sans",
    "axes.grid":        False,
})


# ── FIGURE & LAYOUT ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG)

gs = GridSpec(
    2, 2, figure=fig,
    left=0.07, right=0.97, top=0.87, bottom=0.10,
    hspace=0.45, wspace=0.32,
    height_ratios=[3, 1],
)

ax_hist  = fig.add_subplot(gs[:, 0])   # full-height left
ax_conv  = fig.add_subplot(gs[0, 1])   # top right — convergence
ax_stats = fig.add_subplot(gs[1, 1])   # bottom right — key stats

for ax in (ax_hist, ax_conv, ax_stats):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.8)


# ── HEADER ────────────────────────────────────────────────────────────────────

fig.text(
    0.5, 0.96,
    "NYC CRASH RISK PREDICTOR",
    ha="center", fontsize=21, fontweight="bold",
    color=YELLOW, path_effects=GLOW,
)
fig.text(
    0.5, 0.924,
    f"ZIP {ZIP_CODE}  ·  {DAY_OF_WEEK} {HOUR}:00 – {HOUR+1}:00  ·  "
    f"Monte Carlo  ·  Gamma–Poisson  ·  {N_TRIALS:,} trials",
    ha="center", fontsize=10.5, color=GRAY,
)


# ── HISTOGRAM PANEL ───────────────────────────────────────────────────────────

max_k      = min(int(np.percentile(results, 99.5)), 13)
bins       = np.arange(-0.5, max_k + 1.5, 1)
bin_ctrs   = (bins[:-1] + bins[1:]) / 2
bar_colors = [GREEN] + [RED] * max_k

ax_hist.set_xlim(-0.65, max_k + 0.65)
ax_hist.set_ylim(0, 0.62)
ax_hist.set_xlabel("Crashes in one hour  (k)", fontsize=11)
ax_hist.set_ylabel("Fraction of trials", fontsize=11)
ax_hist.set_title(
    "Distribution of simulated crash counts",
    fontsize=12, pad=10, fontweight="bold", color=WHITE,
)
ax_hist.set_xticks(range(max_k + 1))
ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Subtle horizontal grid
for yv in np.arange(0.1, 0.65, 0.1):
    ax_hist.axhline(yv, color=BORDER, lw=0.5, zorder=0)

bars = ax_hist.bar(
    bin_ctrs, np.zeros(len(bin_ctrs)),
    width=0.82, color=bar_colors,
    edgecolor=BG, linewidth=0.7, alpha=0.88, zorder=3,
)

# Gradient alpha for reds: darker for higher k
for i, bar in enumerate(bars[1:], start=1):
    bar.set_alpha(0.55 + 0.35 * min(i / max_k, 1.0))

ax_hist.legend(
    handles=[
        mpatches.Patch(color=GREEN, label="k = 0  (no crash)", alpha=0.88),
        mpatches.Patch(color=RED,   label="k ≥ 1  (crash)",   alpha=0.88),
    ],
    fontsize=9.5, framealpha=0.15, loc="upper right", labelcolor=WHITE,
)

trial_counter = ax_hist.text(
    0.03, 0.97, "Trials:  0",
    transform=ax_hist.transAxes, ha="left", va="top",
    fontsize=11, color=YELLOW, fontweight="bold",
)

p_live = ax_hist.text(
    0.97, 0.62, "",
    transform=ax_hist.transAxes, ha="right", va="top",
    fontsize=22, color=RED, fontweight="bold",
    path_effects=GLOW,
)

# Posterior info (static)
ax_hist.text(
    0.03, 0.06,
    f"Posterior:  λ ~ Gamma({alpha_post}, {beta_post})\n"
    f"Prior mean: {alpha_post/beta_post:.3f} crashes/hr",
    transform=ax_hist.transAxes, ha="left", va="bottom",
    fontsize=8.5, color=GRAY, linespacing=1.5,
)


# ── CONVERGENCE PANEL ─────────────────────────────────────────────────────────

ax_conv.set_xlim(0, N_TRIALS)
ax_conv.set_ylim(max(0.4, p_crash - 0.30), min(1.0, p_crash + 0.25))
ax_conv.set_xlabel("Number of trials", fontsize=10)
ax_conv.set_ylabel("Running  P(≥1 crash)", fontsize=10)
ax_conv.set_title(
    "Convergence diagnostic — estimate stabilising",
    fontsize=11, pad=8, fontweight="bold", color=WHITE,
)
ax_conv.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_conv.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

for yv in np.linspace(ax_conv.get_ylim()[0], ax_conv.get_ylim()[1], 6):
    ax_conv.axhline(yv, color=BORDER, lw=0.5, zorder=0)

# True estimate dashed line
ax_conv.axhline(
    p_crash, color=RED, lw=1.4, ls="--", alpha=0.65,
    label=f"Converged: {p_crash:.1%}", zorder=2,
)
ax_conv.legend(fontsize=9.5, framealpha=0.15, labelcolor=WHITE, loc="upper right")

# Static ±1 SE funnel (faint)
ax_conv.fill_between(
    running_n,
    running_p - 1.96 * running_se,
    running_p + 1.96 * running_se,
    alpha=0.07, color=BLUE, zorder=1,
)

conv_line, = ax_conv.plot([], [], color=BLUE, lw=2.0, alpha=0.95, zorder=3)


# ── STATS PANEL ───────────────────────────────────────────────────────────────

ax_stats.axis("off")

# Top divider
ax_stats.plot([0, 1], [1.02, 1.02], color=BORDER, lw=1.0,
              transform=ax_stats.transAxes, clip_on=False)

stat_data = [
    ("P(≥1 crash)",   f"{p_crash:.1%}",                          RED),
    ("Mean crashes",  f"{mean_crashes:.3f}",                      BLUE),
    ("95 % CI",       f"[{int(ci_low)}, {int(ci_high)}]",        PURPLE),
    ("Posterior  λ",  f"{alpha_post / beta_post:.3f}  /hr",       YELLOW),
]

label_artists, value_artists = [], []
for j, (lbl, val, clr) in enumerate(stat_data):
    x = 0.04 + j * 0.255
    tl = ax_stats.text(
        x, 0.72, lbl, transform=ax_stats.transAxes,
        fontsize=9, color=GRAY, ha="left", alpha=0.0,
    )
    tv = ax_stats.text(
        x, 0.05, val, transform=ax_stats.transAxes,
        fontsize=16, color=clr, fontweight="bold", ha="left", alpha=0.0,
    )
    label_artists.append(tl)
    value_artists.append(tv)


# ── PROGRESS BAR (thin strip) ─────────────────────────────────────────────────

ax_prog = fig.add_axes([0.07, 0.035, 0.90, 0.016])
ax_prog.set_facecolor(BG)
ax_prog.axis("off")
ax_prog.set_xlim(0, 1)
ax_prog.set_ylim(0, 1)

ax_prog.barh(0.5, 1.0, height=1.0, color=BORDER, left=0, align="center")
prog_bar = ax_prog.barh(0.5, 0.0, height=1.0, color=BLUE, left=0, align="center")

prog_label = ax_prog.text(
    0.5, 0.5, "", ha="center", va="center",
    fontsize=7.5, color=WHITE, fontweight="bold",
    transform=ax_prog.transAxes,
)


# ── ANIMATION FRAME SCHEDULE ─────────────────────────────────────────────────
# Log-spaced so early trials (noisy, interesting) get more screen time.
# Final 15 frames hold on N_TRIALS so stats can fade in cleanly.

TOTAL_FRAMES  = 150
HOLD_FRAMES   = 15
ACTIVE_FRAMES = TOTAL_FRAMES - HOLD_FRAMES

frame_ns = np.unique(
    np.logspace(1, np.log10(N_TRIALS), ACTIVE_FRAMES).astype(int)
).clip(1, N_TRIALS)

# Pad / trim to exactly ACTIVE_FRAMES
while len(frame_ns) < ACTIVE_FRAMES:
    frame_ns = np.append(frame_ns, N_TRIALS)
frame_ns = frame_ns[:ACTIVE_FRAMES]

# Append hold frames
frame_ns = np.concatenate([frame_ns, np.full(HOLD_FRAMES, N_TRIALS)])


# ── UPDATE FUNCTION ───────────────────────────────────────────────────────────

def update(fi):
    n = int(frame_ns[fi])

    # — Histogram —
    raw, _ = np.histogram(results[:n], bins=bins)
    fracs  = raw / max(n, 1)
    for bar, frac in zip(bars, fracs):
        bar.set_height(frac)

    trial_counter.set_text(f"Trials:  {n:,}")

    cur_p = (results[:n] >= 1).mean()
    p_live.set_text(f"P(crash)\n{cur_p:.1%}")

    # — Convergence line —
    conv_line.set_data(np.arange(1, n + 1), running_p[:n])

    # — Progress bar —
    prog_bar[0].set_width(n / N_TRIALS)
    prog_label.set_text(f"{n:,} / {N_TRIALS:,} trials  —  {n/N_TRIALS:.0%}")

    # — Stats fade-in (last HOLD_FRAMES + 5 frames) —
    fade_start = TOTAL_FRAMES - HOLD_FRAMES - 5
    if fi >= fade_start:
        alpha = min(1.0, (fi - fade_start) / (HOLD_FRAMES * 0.8))
        for t in label_artists + value_artists:
            t.set_alpha(alpha)

    return (
        [*bars, conv_line, trial_counter, p_live,
         prog_bar[0], prog_label,
         *label_artists, *value_artists]
    )


# ── RUN ANIMATION ─────────────────────────────────────────────────────────────

ani = animation.FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    interval=55,          # ~18 fps
    blit=True,
    repeat=False,
)

# ── SAVE ─────────────────────────────────────────────────────────────────────

# GIF (no extra dependencies)
gif_path = "simulation_animated.gif"
print(f"Saving GIF → {gif_path}  (may take 30–60 s …)")
ani.save(gif_path, writer="pillow", fps=18, dpi=110)
print(f"  ✓  {gif_path}")

# Optional MP4 — uncomment if ffmpeg is installed
# mp4_path = "simulation_animated.mp4"
# print(f"Saving MP4 → {mp4_path} …")
# ani.save(mp4_path, writer="ffmpeg", fps=18, dpi=150,
#          extra_args=["-vcodec", "libx264", "-crf", "18"])
# print(f"  ✓  {mp4_path}")

# Static PNG — final frame for slides / thumbnail
update(TOTAL_FRAMES - 1)
fig.savefig("simulation_output.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("  ✓  simulation_output.png")

plt.show()
