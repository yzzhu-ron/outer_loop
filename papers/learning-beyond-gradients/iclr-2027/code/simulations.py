"""Numerical illustrations for "Learning Beyond Gradients" (ICLR 2027 submission, v2).

Three simulations, one per theorem family:
  1. Ratchet illusion under evaluation noise (greedy single-eval ratchet)
     + repeat-gated variant, with a log-scale inset showing the gated
     false-accept rate honestly (v2: addresses reviewer items N4/W9-iv).
  2. Feedback-channel separation. v2: ALL THREE arms now execute a real
     algorithm (addresses W7/N4): scalar = uniform search without
     replacement over an explicit random permutation; first-error = the
     prefix-fixing algorithm; distance = the L+1 probing scheme with a
     correctness assertion on the reconstructed target.
  3. Deceptive landscape, tolerant greedy vs. archive search. v2: the
     ascent slope is asymmetric, f(i) = 2(i-k) on [k, 2k], so that 2k is
     the UNIQUE global optimum (addresses W3/E1). Acceptance dynamics are
     unchanged (right moves on the ascent are still strict improvements),
     so the exact birth-death expectation is identical to v1.

Outputs vector PDFs into ../figures/. Deterministic (seeded). Runtime < 1 min.
Note: the RNG call order of simulation 1 is kept byte-identical to v1 so
that the reviewer-verified seed-0 numbers (median best 3.39 sigma at
t = 2000; 2 gated false accepts across 400 x 2000 decisions) reproduce.
"""

import math

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


def normal_tail(x):
    """P(Z >= x) for standard normal Z, exact via erfc."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ----------------------------------------------------------------------------
# Simulation 1: the ratchet illusion.
# Null edits (true gain 0), measured score = N(0, sigma^2).
# Greedy single-eval ratchet: accept iff measured > running max.
# ----------------------------------------------------------------------------
def sim_ratchet(T=2000, R=400, sigma=1.0):
    Z = rng.normal(0.0, sigma, size=(R, T))
    running_max = np.maximum.accumulate(Z, axis=1)
    accepts = np.cumsum(Z == running_max, axis=1)  # records of an iid sequence

    # Repeat-gated ratchet (one-sided gate, Corollary cor:budget(ii)):
    # N = ceil((8 sigma^2/Delta^2) ln(B/delta)) evals per candidate against an
    # incumbent of KNOWN mean (here 0); accept iff candidate mean >= Delta/2,
    # with Delta = 1, delta = .05.
    delta_gap, delta_conf = 1.0, 0.05
    N = int(np.ceil(8 * sigma**2 / delta_gap**2 * np.log(T / delta_conf)))
    means = rng.normal(0.0, sigma / np.sqrt(N), size=(R, T))
    gated_accepts = np.cumsum(means >= delta_gap / 2, axis=1)

    # Exact per-candidate false-accept probability and Poisson prediction.
    p_gate = normal_tail(0.5 * np.sqrt(N) / sigma * delta_gap)
    total_gated = int(gated_accepts[:, -1].sum())
    med_best = float(np.median(running_max[:, -1]))
    mean_best = float(np.mean(running_max[:, -1]))
    mean_accepts = float(accepts[:, -1].mean())

    print(f"sim1: N gate = {N}, per-candidate p = {p_gate:.3g}, "
          f"expected total over {R}x{T} = {p_gate * R * T:.2f}, "
          f"observed total = {total_gated}")
    print(f"sim1: best at t={T}: median {med_best:.3f} sigma, "
          f"mean {mean_best:.3f} sigma "
          f"(asymptotic ceiling sqrt(2 ln t) = {np.sqrt(2 * np.log(T)):.3f})")
    print(f"sim1: mean single-eval accepts at t={T}: {mean_accepts:.2f} "
          f"(H_t = {np.sum(1.0 / np.arange(1, T + 1)):.2f})")

    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 1.95))

    ax = axes[0]
    med = np.median(running_max, axis=0)
    lo, hi = np.percentile(running_max, [10, 90], axis=0)
    ax.fill_between(t, lo, hi, color=C_BLUE, alpha=0.18, lw=0)
    ax.plot(t, med, color=C_BLUE, label="reported best (greedy ratchet)")
    ax.plot(t, sigma * np.sqrt(2 * np.log(t)), "--", color=C_RED,
            label=r"$\sigma\sqrt{2\ln t}$ upper bound (Thm. 5.1)")
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
            label=fr"false accepts, $N={N}$ gate")
    ax.set_xlabel(r"candidate edits evaluated $t$")
    ax.set_ylabel("accepted null edits")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Records accumulate like $\\ln t$")

    # v2 inset (log y): the gated rate is small but NOT exactly zero --
    # 2 accepts across 8x10^5 decisions for this seed, matching the
    # Poisson prediction p*t per run. Shown honestly instead of an
    # indistinguishable-from-the-axis flat line.
    axins = ax.inset_axes([0.56, 0.12, 0.41, 0.36])
    axins.plot(t, gated_accepts.mean(axis=0), color=C_GREEN, lw=1.0)
    axins.plot(t, p_gate * t, ":", color=C_GRAY, lw=1.0,
               label=r"$\mathbb{E}=p_N t$")
    axins.set_yscale("log")
    axins.set_ylim(1e-4, 3e-2)
    axins.set_xscale("log")
    axins.tick_params(labelsize=5)
    axins.set_title("gated rate, log scale", fontsize=5.5, pad=2)
    axins.legend(frameon=False, fontsize=5, loc="upper left")

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGDIR}/fig_ratchet.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"sim1 done (N gate = {N})")


# ----------------------------------------------------------------------------
# Simulation 2: feedback-channel separation on the L-bit identification game.
# v2: every arm executes its algorithm end to end.
# ----------------------------------------------------------------------------
def sim_feedback():
    Ls = np.arange(2, 15)
    reps = 60

    # Scalar 0/1 reward: uniform search without replacement, executed over an
    # explicit random permutation of all 2^L candidate indices.
    scalar_mean = []
    for L in Ls:
        n = 2**L
        counts = []
        for _ in range(reps):
            target = int(rng.integers(0, n))
            order = rng.permutation(n)
            q = 0
            for cand in order:
                q += 1
                if cand == target:  # feedback 1{a = a*}
                    break
            counts.append(q)
        scalar_mean.append(np.mean(counts))

    # First-error feedback: prefix-fixing algorithm of Thm 4.2(ii).
    # The count includes the final CONFIRMING query (the one whose feedback
    # is "no error"), so the realized mean is ~ L/2 + 1 for a uniform
    # target; identification itself uses at most L queries (see App. G).
    firsterr_mean = []
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
        firsterr_mean.append(np.mean(counts))

    # Distance feedback: the probing scheme of Thm 4.2(iii), executed.
    # Query the all-zeros base, then flip each bit once; bit i of the target
    # is 1 iff the distance drops. The reconstruction is asserted correct.
    hamming_mean = []
    for L in Ls:
        counts = []
        for _ in range(reps):
            target = rng.integers(0, 2, size=L)
            a0 = np.zeros(L, dtype=int)
            q = 1                                   # base query
            d0 = int(np.sum(a0 != target))
            guess = np.zeros(L, dtype=int)
            for i in range(L):
                probe = a0.copy()
                probe[i] ^= 1
                q += 1
                d_i = int(np.sum(probe != target))
                guess[i] = 1 if d_i == d0 - 1 else 0
            assert np.array_equal(guess, target), "probing failed to identify"
            counts.append(q)                        # = L + 1, by construction
        hamming_mean.append(np.mean(counts))

    print(f"sim2: scalar means {np.round(scalar_mean, 1).tolist()}")
    print(f"sim2: first-error means {np.round(firsterr_mean, 2).tolist()} "
          f"(law L/2+1: {[(L / 2 + 1) for L in Ls]})")
    print(f"sim2: distance means {np.round(hamming_mean, 1).tolist()} (= L+1)")

    Lgrid = np.linspace(2, 14, 100)
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ax.plot(Ls, scalar_mean, "o", ms=3, color=C_RED)
    ax.plot(Lgrid, (2**Lgrid + 1) / 2, "--", color=C_RED,
            label=r"0/1 reward, mean law $\frac{1}{2}(2^L\!+\!1)$ (Thm. 4.2i)")
    ax.plot(Ls, hamming_mean, "s", ms=3, color=C_ORANGE)
    ax.plot(Lgrid, Lgrid + 1, "--", color=C_ORANGE,
            label=r"distance probing, $L+1$ (Thm. 4.2iii)")
    ax.plot(Ls, firsterr_mean, "^", ms=3, color=C_GREEN)
    ax.plot(Lgrid, Lgrid / 2 + 1, "--", color=C_GREEN,
            label=r"first-error, average case $L/2+1$")
    ax.set_yscale("log")
    ax.set_xlabel(r"artifact description length $L$ (bits)")
    ax.set_ylabel("queries (incl. confirmation)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGDIR}/fig_feedback.pdf", bbox_inches="tight")
    plt.close(fig)
    print("sim2 done")


# ----------------------------------------------------------------------------
# Simulation 3: deceptive valley. epsilon-tolerant greedy vs. archive search.
# Chain 0..2k. v2 landscape (unique optimum):
#   f(i) = k - i        on [0, k]
#   f(i) = 2*(i - k)    on [k, 2k]      so f(2k) = 2k > k = f(0).
# Proposal convention (stated in Thm 5.4): propose a uniformly random
# element of {i-1, i+1}; out-of-range proposals are rejected (walk stays).
# ----------------------------------------------------------------------------
def f_landscape(i, k):
    return (k - i) if i <= k else 2 * (i - k)


def exact_eps_greedy_time(k, eps):
    """Exact E[T] from 0 to 2k for the tolerant-greedy birth-death chain.

    Identical to v1: the asymmetric ascent slope changes f-values, not
    accept/reject decisions (right moves on the ascent remain strict
    improvements), so the transition probabilities are unchanged.
    """
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
            continue  # out-of-range proposal rejected; walk stays
        if f_landscape(nxt, k) > f_landscape(pos, k) or rng.random() < eps:
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

    # Verification printout for App. G.
    ratio = exact[-1] / (2 * eps ** (-ks_exact[-1]))
    print(f"sim3: exact E[T] / lower bound 2 eps^-k at k=25: {ratio:.2f}")
    for kk in (5, 10, 25):
        a = arch[list(ks_arch).index(kk)]
        print(f"sim3: archive mean at k={kk}: {a:.0f} "
              f"(bound 4k(2k+1) = {4 * kk * (2 * kk + 1)})")

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
