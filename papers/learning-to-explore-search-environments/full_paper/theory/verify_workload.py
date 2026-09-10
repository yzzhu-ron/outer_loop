"""Write exact constructed witnesses; never reads retrieval outcomes."""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import workload_design as design
from finite_decisions import environment_information


def main():
    root = Path(__file__).resolve().parent
    utility, prior, channels = design.bit_problem(2, 3)
    rows = design.risk_matrix(utility, prior, channels)
    robust = design.minimax_design(rows, ((1, 0), (0, 1)))
    nominal = design.minimax_design(rows, ((F(9, 10), F(1, 10)),))
    result = {
        'status': 'Exact constructed finite-model witnesses; not empirical retrieval results or a novelty claim.',
        'contract': 'Nature selects a fixed workload before randomized onboarding; query context is observed when routing.',
        'two_task_types_three_nuisance_bits': {
            'worlds': len(utility), 'prior_risks': design.risks(utility, prior),
            'context_by_design_residual_risk': rows,
            'design_names': ['reveal_context_0_bit', 'reveal_context_1_bit', 'reveal_three_nuisance_bits'],
            'information_bits': [environment_information(prior, channel) for channel in channels],
            'robust_design': asdict(robust), 'nominal_90_10_design': asdict(nominal),
            'nominal_design_shifted_to_context_1_risk': sum(p * value for p, value in zip(nominal.mixture, rows[1])),
            'uniform_random_design_worst_risk': max(sum(row) / len(row) for row in rows),
            'universal_optimum': design.universally_best(rows),
            'context_crossing_witnesses': design.crossing_pairs(rows),
        },
        'workload_adversary_timing': design.robust_conditioning_example(),
        'five_contexts_side_information_tradeoff': [
            {'budget': budget, 'available_messages': messages,
             'exact_minimax_risk': design.timing_risk(5, budget, messages)}
            for messages in range(1, 6) for budget in range(6)
        ],
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (root / 'workload_design.py', root / 'verify_workload.py',
                           root.parents[1] / 'theory' / 'finite_decisions.py')},
    }
    output = root / 'verified_workload_results.json'
    output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + '\n')
    print(output)


if __name__ == '__main__':
    main()
