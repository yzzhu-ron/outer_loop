"""Numerical illustrations for "Learning Beyond Gradients" (ICML 2026 submission).

Three simulations, one per theorem family:
  1. Ratchet illusion under evaluation noise (Thm: greedy single-eval ratchet).
  2. Feedback-channel separation (Thm: scalar reward vs. structured feedback).
  3. Deceptive landscape: tolerant greedy vs. archive search (Thm: deception).

Outputs vector PDFs into ../figures/. Deterministic (seeded). Runtime < 30 s.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
})

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#777777")

FIGDIR = "../figures"


# ----------------------------------------------------------------------------
# Simulation 1: the ratchet illusion.
# Null edits (true gain 0), measured score = N(0, sigma^2).
# Greedy single-eval ratchet: accept iff measured > running max.
# ----------------------------------------------------------------------------
def sim_ratchet(T=2000, R=400, sigma=1.0):
    Z = rng.normal(0.0, sigma, size=(R, T))
    running_max = np.maximum.accumulate(Z, axis=1)
    accepts = np.cumsum(Z == running_max, axis=1)  # records of an iid sequence

    # Repeat-gated ratchet (Corollary): N = ceil((8 sigma^2/Delta^2) ln(B/delta))
    # evals per candidate, accept iff mean >= Delta/2, with Delta = 1, delta = .05.
    delta_gap, delta_conf = 1.0, 0.05
    N = int(np.ceil(8 * sigma**2 / delta_gap**2 * np.log(T / delta_conf)))
    means = rng.normal(0.0, sigma / np.sqrt(N), size=(R, T))
    gated_accepts = np.cumsum(means >= delta_gap / 2, axis=1)

    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 1.95))

    ax = axes[0]
    med = np.median(running_max, axis=0)
    lo, hi = np.percentile(running_max, [10, 90], axis=0)
    ax.fill_between(t, lo, hi, color=C_BLUE, alpha=0.18, lw=0)
    ax.plot(t, med, color=C_BLUE, label="reported best (greedy ratchet)")
    ax.plot(t, sigma * np.sqrt(2 * np.log(t)), "--", color=C_RED,
            label=r"$\sigma\sqrt{2\ln t}$ (Thm. 5.1)")
    ax.axhline(0.0, color=C_GRAY, lw=1.0, label="true value of every artifact")
    ax.set_xlabel(r"candidate edits evaluated $t$")
    ax.set_ylabel(r"score (units of $\sigma$)")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Improvement that is not there")

    ax = axes[1]
    ax.plot(t, accepts.mean(axis=0), color=C_BLUE,
            label="false accepts, single eval")
    ax.plot(t, np.cumsum(1.0 / t), "--", color=C_RED,
            label=r"$H_t=\sum_{i\leq t}1/i$ (Thm. 5.1ii)")
    ax.plot(t, gated_accepts.mean(axis=0), color=C_GREEN,
            label=fr"false accepts, $N={N}$ repeats (Cor. 5.3)")
    ax.set_xlabel(r"candidate edits evaluated $t$")
    ax.set_ylabel("accepted null edits")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Records accumulate like $\\ln t$")

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGDIR}/fig_ratchet.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"sim1 done (N gate = {N})")


# ----------------------------------------------------------------------------
# Simulation 2: feedback-channel separation on the L-bit identification game.
# ----------------------------------------------------------------------------
def sim_feedback():
    Ls = np.arange(2, 15)
    reps = 60

    scalar_med = []
    for L in Ls:
        n = 2**L
        # uniform search without replacement for a uniform needle:
        # queries = position of needle in a random permutation
        qs = rng.integers(1, n + 1, size=reps)
        scalar_med.append(np.mean(qs))

    firsterr = []
    for L in Ls:
        counts = []
        for _ in range(reps):
            target = rng.integers(0, 2, size=L)
            a = np.zeros(L, dtype=int)
            q = 0
            while True:
                q += 1
                diff = np.nonzero(a != target)[0]
                if diff.size == 0:
                    break
                a[diff[0]] ^= 1  # feedback: index of first wrong bit
            counts.append(q)
        firsterr.append(np.mean(counts))

    hamming = []
    for L in Ls:
        # probe scheme: base query + flip each bit once, keep flips that
        # reduce Hamming distance -> exactly L+1 queries
        hamming.append(L + 1)

    Lgrid = np.linspace(2, 14, 100)
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ax.plot(Ls, scalar_med, "o", ms=3, color=C_RED)
    ax.plot(Lgrid, (2**Lgrid + 1) / 2, "--", color=C_RED,
            label=r"0/1 reward: $\frac{1}{2}(2^L\!+\!1)$ (Thm. 4.2)")
    ax.plot(Ls, hamming, "s", ms=3, color=C_ORANGE)
    ax.plot(Lgrid, Lgrid + 1, "--", color=C_ORANGE,
            label=r"distance feedback: $L+1$")
    ax.plot(Ls, firsterr, "^", ms=3, color=C_GREEN)
    ax.plot(Lgrid, Lgrid / 2 + 1, "--", color=C_GREEN,
            label=r"first-error feedback: $\leq L$ (Thm. 4.2)")
    ax.set_yscale("log")
    ax.set_xlabel(r"artifact description length $L$ (bits)")
    ax.set_ylabel("queries to identify $a^*$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGDIR}/fig_feedback.pdf", bbox_inches="tight")
    plt.close(fig)
    print("sim2 done")


# ----------------------------------------------------------------------------
# Simulation 3: deceptive valley. epsilon-tolerant greedy vs. archive search.
# Chain 0..2k, f decreasing on [0,k], increasing on [k,2k], f(2k) maximal.
# ----------------------------------------------------------------------------
def exact_eps_greedy_time(k, eps):
    """Exact E[T] from 0 to 2k for the tolerant-greedy birth-death chain."""
    # state i in [0, 2k]; p_i = P(move right), q_i = P(move left)
    total = 0.0
    h_prev = 0.0
    for i in range(0, 2 * k):
        if i == 0:
            p, q = eps / 2.0, 0.0
        elif i < k:           # valley: right worsens, left improves
            p, q = eps / 2.0, 0.5
        elif i == k:          # valley floor: both neighbors improve
            p, q = 0.5, 0.5
        else:                 # ascent: right improves, left worsens
            p, q = 0.5, eps / 2.0
        h = (1.0 + q * h_prev) / p
        total += h
        h_prev = h
    return total


def sim_eps_greedy(k, eps, cap=4_000_000):
    pos, t = 0, 0
    while pos != 2 * k and t < cap:
        t += 1
        step = 1 if rng.random() < 0.5 else -1
        nxt = pos + step
        if nxt < 0 or nxt > 2 * k:
            continue
        f_cur = (k - pos) if pos <= k else (pos - k)
        f_nxt = (k - nxt) if nxt <= k else (nxt - k)
        if f_nxt > f_cur or rng.random() < eps:
            pos = nxt
    return t


def sim_archive(k, reps=60):
    ts = []
    for _ in range(reps):
        archive = [0]
        in_archive = {0}
        t = 0
        while 2 * k not in in_archive:
            t += 1
            x = archive[rng.integers(len(archive))]
            y = x + (1 if rng.random() < 0.5 else -1)
            if 0 <= y <= 2 * k and y not in in_archive:
                in_archive.add(y)
                archive.append(y)
        ts.append(t)
    return np.mean(ts)


def sim_deception(eps=0.3):
    ks_exact = np.arange(2, 26)
    exact = [exact_eps_greedy_time(k, eps) for k in ks_exact]

    ks_sim = np.arange(2, 11)
    sim = [np.mean([sim_eps_greedy(k, eps) for _ in range(12)])
           for k in ks_sim]

    ks_arch = np.arange(2, 26)
    arch = [sim_archive(k) for k in ks_arch]

    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ax.plot(ks_exact, exact, "--", color=C_RED,
            label=fr"tolerant greedy, exact $\mathbb{{E}}[T]$ ($\epsilon={eps}$)")
    ax.plot(ks_sim, sim, "o", ms=3, color=C_RED, label="tolerant greedy, simulated")
    ax.plot(ks_arch, arch, "s-", ms=3, color=C_GREEN,
            label="archive (novelty) search")
    ax.plot(ks_arch, 4.0 * ks_arch * (2 * ks_arch + 1), ":", color=C_GRAY,
            label=r"$4k(2k\!+\!1)$ bound (Thm. 5.4iii)")
    ax.set_yscale("log")
    ax.set_xlabel(r"valley half-width $k$")
    ax.set_ylabel(r"steps to reach optimum")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGDIR}/fig_deception.pdf", bbox_inches="tight")
    plt.close(fig)
    print("sim3 done")


if __name__ == "__main__":
    sim_ratchet()
    sim_feedback()
    sim_deception()
    print("all figures written to", FIGDIR)
