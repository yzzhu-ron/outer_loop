"""Reproduce exact constructions: python theory/verify_theory.py."""

from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

from finite_decisions import (
    bayes_regret,
    deterministic_likelihood,
    environment_information,
    expected_regret_after,
    gaussian_variance_reduction,
    minimax_radius,
    pairwise_incompatibility,
    probe_value,
    robust_after_partition,
    select_action_from_contrasts,
)


def run():
    reverse = ((1, 0), (0, 1))
    cyclic = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
    cyclic_radius = minimax_radius(cyclic)
    epsilon = F(1, 100)
    strict_cyclic = ((0, 1, 1 - epsilon), (1 - epsilon, 0, 1), (1, 1 - epsilon, 0))
    worlds = [(decision, nuisance) for decision in range(2) for nuisance in range(8)]
    utilities = tuple((int(d == 0), int(d == 1)) for d, _ in worlds)
    prior = (F(1, len(worlds)),) * len(worlds)
    candidates = {
        "hard_independent": ((F(1, 2), F(1, 2)),) * len(worlds),
        "easy_aligned": tuple((F(1, 5), F(4, 5)) if d == 0 else (F(0), F(1)) for d, _ in worlds),
        "nuisance_signature": tuple(
            tuple((F(1, 20) if hit == 0 else F(19, 20)) if symbol == n else F(0) for symbol in range(8) for hit in range(2))
            for _, n in worlds
        ),
    }
    # These hit probabilities match the joint observation laws above. The
    # nuisance observation reveals an ancillary symbol plus an independent
    # Bernoulli(19/20) hit. No real corpus result is claimed.
    rediscovery = {"hard_independent": F(1, 2), "easy_aligned": F(9, 10), "nuisance_signature": F(19, 20)}
    candidate_results = {
        name: {
            "expected_rediscovery": rediscovery[name],
            "environment_information_bits": environment_information(prior, likelihood),
            "regret_reduction": probe_value(utilities, prior, likelihood),
            "downstream_regret_after_one_probe": expected_regret_after(utilities, prior, likelihood),
        }
        for name, likelihood in candidates.items()
    }
    selection = {
        "hardest": min(rediscovery, key=rediscovery.get),
        "environment_entropy": max(candidate_results, key=lambda q: candidate_results[q]["environment_information_bits"]),
        "decision_value": max(candidate_results, key=lambda q: candidate_results[q]["regret_reduction"]),
    }
    xor_worlds = [(a, b) for a in range(2) for b in range(2)]
    xor_utilities = [(int((a ^ b) == 0), int((a ^ b) == 1)) for a, b in xor_worlds]
    xor_prior = (F(1, 4),) * 4
    xor_single = [probe_value(xor_utilities, xor_prior, deterministic_likelihood([w[i] for w in xor_worlds])) for i in range(2)]
    # T=(0,1)^2 has interior. V=(1,0), W=(0,1), b=-1/2:
    # changing theta_2 leaves the probe mean unchanged but reverses the winner.
    interior_points = ((F(1, 2), F(1, 4)), (F(1, 2), F(3, 4)))
    interior_contrasts = tuple(point[1] - F(1, 2) for point in interior_points)
    return {
        "provenance": "Exact rational finite constructions; no retrieval data, Monte Carlo, fitting, or empirical generalization claims.",
        "identical_perfect_rediscovery": {
            "known_document_success": 1,
            "robust_regret": robust_after_partition(reverse, [1, 1]),
            "regret_after_aligned_revealing_probe": robust_after_partition(reverse, [0, 1]),
        },
        "three_way_incompatibility": {
            "utilities": cyclic,
            "pairwise_incompatibility": pairwise_incompatibility(cyclic),
            "robust_regret": cyclic_radius.value,
            "optimal_action_mixture": cyclic_radius.action_mixture,
            "worst_prior": cyclic_radius.witness_prior,
        },
        "three_way_unique_winners": {
            "epsilon": epsilon,
            "utilities": strict_cyclic,
            "pairwise_incompatibility": pairwise_incompatibility(strict_cyclic),
            "robust_regret": minimax_radius(strict_cyclic).value,
        },
        "one_probe_acquisition": {
            "prior_downstream_regret": bayes_regret(utilities, prior),
            "candidates": candidate_results,
            "selected": selection,
            "selected_regret": {method: candidate_results[q]["downstream_regret_after_one_probe"] for method, q in selection.items()},
            "uniform_random_expected_regret": sum(row["downstream_regret_after_one_probe"] for row in candidate_results.values()) / len(candidate_results),
        },
        "complementarity": {
            "one_step_values": xor_single,
            "two_probe_value": probe_value(xor_utilities, xor_prior, deterministic_likelihood(xor_worlds)),
        },
        "gaussian_action_alignment": {
            "prior_covariance": ((100, 0), (0, 1)),
            "task_contrast": (0, 1),
            "unit_noise_nuisance_probe_variance_reduction": gaussian_variance_reduction(((100, 0), (0, 1)), (1, 0), (0, 1), 1),
            "unit_noise_aligned_probe_variance_reduction": gaussian_variance_reduction(((100, 0), (0, 1)), (0, 1), (0, 1), 1),
        },
        "reference_router_alignment": {
            "utilities": (1, 0),
            "perfect_nonreference_contrast": -1,
            "selected_action_including_reference": select_action_from_contrasts((-1,)),
            "regret_including_reference": 0,
            "regret_if_reference_is_omitted": 1,
        },
        "bounded_domain_nonidentifiability": {
            "admissible_domain": "T=(0,1)^2; utility of reference=1/2, alternative=theta_2",
            "interior_parameter_points": interior_points,
            "probe_means": tuple(point[0] for point in interior_points),
            "nonreference_contrasts": interior_contrasts,
            "optimal_actions": tuple(select_action_from_contrasts((g,)) for g in interior_contrasts),
        },
    }


if __name__ == "__main__":
    results = run()
    destination = Path(__file__).with_name("verified_results.json")
    destination.write_text(json.dumps(results, indent=2, default=lambda x: str(x) if isinstance(x, F) else x) + "\n")
    print(destination)
    print("Exact constructions reproduced; rational values are stored as strings.")
