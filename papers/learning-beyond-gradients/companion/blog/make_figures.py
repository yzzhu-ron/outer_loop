"""Regenerate the three blog figures into assets/. Seeded, < 1 min, NumPy + Matplotlib only."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
os.makedirs("assets", exist_ok=True)

CLAY, OLIVE, SKY, SLATE, GRAY = "#D97757", "#788C5D", "#6A8CAF", "#141413", "#87867F"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.size": 11,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "axes.linewidth": 0.8,
    "lines.linewidth": 1.8, "figure.dpi": 170, "savefig.facecolor": "#FAF9F5",
    "axes.facecolor": "#FAF9F5", "axes.spines.top": False, "axes.spines.right": False,
})


def fig_casino(T=2000, R=400, sigma=1.0):
    """The ratchet illusion: pure-noise edits, greedy single-eval acceptance."""
    Z = rng.normal(0, sigma, (R, T))
    best = np.maximum.accumulate(Z, axis=1)
    accepts = np.cumsum(Z == best, axis=1)
    N = int(np.ceil(8 * np.log(T / 0.05)))                      # repeat gate (N=85)
    gated = np.cumsum(rng.normal(0, sigma / np.sqrt(N), (R, T)) >= 0.5, axis=1)

    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1))
    lo, hi = np.percentile(best, [10, 90], axis=0)
    ax[0].fill_between(t, lo, hi, color=CLAY, alpha=0.15, lw=0)
    ax[0].plot(t, np.median(best, axis=0), color=CLAY, label="reported best (greedy ratchet)")
    ax[0].plot(t, sigma * np.sqrt(2 * np.log(t)), "--", color=SLATE, lw=1.3,
               label=r"$\sigma\sqrt{2\ln t}$")
    ax[0].axhline(0, color=GRAY, lw=1, label="true value of every edit")
    ax[0].set(xscale="log", xlabel="edits evaluated $t$", ylabel=r"score ($\sigma$ units)",
              title="(a) improvement that is not there")
    ax[0].legend(frameon=False, loc="upper left")
    ax[1].plot(t, accepts.mean(0), color=CLAY, label="false accepts, single eval")
    ax[1].plot(t, np.cumsum(1 / t), "--", color=SLATE, lw=1.3, label=r"$H_t \approx \ln t$")
    ax[1].plot(t, gated.mean(0), color=OLIVE, label=f"false accepts, $N={N}$ repeats")
    ax[1].set(xscale="log", xlabel="edits evaluated $t$", ylabel="accepted null edits",
              title="(b) the fix: repeat, then accept")
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.6)
    fig.savefig("assets/fig_casino.png", bbox_inches="tight")
    plt.close(fig)


def fig_feedback(reps=60):
    """Guess-the-password: scalar reward vs structured feedback."""
    Ls = np.arange(2, 15)
    scalar = [np.mean(rng.integers(1, 2**L + 1, reps)) for L in Ls]   # hit position
    firsterr = []
    for L in Ls:                                                       # real algorithm
        c = []
        for _ in range(reps):
            target, a, q = rng.integers(0, 2, L), np.zeros(L, int), 0
            while True:
                q += 1
                d = np.nonzero(a != target)[0]
                if d.size == 0:
                    break
                a[d[0]] ^= 1
            c.append(q)
        firsterr.append(np.mean(c))
    g = np.linspace(2, 14, 100)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(Ls, scalar, "o", ms=5, color=CLAY)
    ax.plot(g, (2**g + 1) / 2, "--", color=CLAY, label=r"scalar reward: $\sim 2^L$")
    ax.plot(g, g + 1, "--", color=SKY, label=r"distance feedback: $L+1$ (exact)")
    ax.plot(Ls, firsterr, "^", ms=5, color=OLIVE)
    ax.plot(g, g / 2 + 1, "--", color=OLIVE, label=r"first-error feedback: $\leq L$")
    ax.set(yscale="log", xlabel="artifact length $L$ (bits)", ylabel=r"queries to identify $a^*$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.6)
    fig.savefig("assets/fig_feedback.png", bbox_inches="tight")
    plt.close(fig)


def fig_valley(eps=0.3):
    """Deceptive valley: tolerant greedy (exact E[T]) vs archive search (simulated)."""
    def exact_greedy(k):                       # birth-death first-step recurrences
        total, h = 0.0, 0.0
        for i in range(2 * k):
            p, q = ((eps / 2, 0.0) if i == 0 else (eps / 2, 0.5) if i < k
                    else (0.5, 0.5) if i == k else (0.5, eps / 2))
            h = (1 + q * h) / p
            total += h
        return total

    def sim_archive(k, reps=50):
        ts = []
        for _ in range(reps):
            arch, seen, t = [0], {0}, 0
            while 2 * k not in seen:
                t += 1
                y = arch[rng.integers(len(arch))] + (1 if rng.random() < 0.5 else -1)
                if 0 <= y <= 2 * k and y not in seen:
                    seen.add(y); arch.append(y)
            ts.append(t)
        return np.mean(ts)

    ks = np.arange(2, 26)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ks, [exact_greedy(k) for k in ks], "--", color=CLAY,
            label=fr"tolerant greedy ($\epsilon={eps}$): $\sim\epsilon^{{-k}}$")
    ax.plot(ks, [sim_archive(k) for k in ks], "s-", ms=4, color=OLIVE,
            label="archive (novelty) search, simulated")
    ax.plot(ks, 4 * ks * (2 * ks + 1), ":", color=GRAY, label=r"$4k(2k+1)$ bound")
    ax.set(yscale="log", xlabel="valley half-width $k$", ylabel="steps to reach the optimum")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.6)
    fig.savefig("assets/fig_valley.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_casino(); print("fig_casino.png")
    fig_feedback(); print("fig_feedback.png")
    fig_valley(); print("fig_valley.png")
