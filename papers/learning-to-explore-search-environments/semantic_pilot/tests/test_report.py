"""Fail-closed report assembly checks using only temporary synthetic fixtures."""

import copy
import csv
import hashlib
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_report as report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.results.mkdir()
        (self.root / 'cache').mkdir()
        self.environments = [f'{corpus}_{backend}' for corpus in ('alpha', 'beta', 'gamma') for backend in ('bm25', 'dense')]
        methods = ('source_router', 'random', 'fixed', 'information_gain', 'learned_value')
        seeds, budgets = (11, 23, 47), (0, 4, 8, 16, 32)
        protocol = {'datasets': {name: {} for name in ('alpha', 'beta', 'gamma')},
                    'probes': {'seeds': seeds, 'pair_budgets': budgets},
                    'query_selection': {'test_counts': {name: 2 for name in ('alpha', 'beta', 'gamma')}}}
        write_json(self.root / 'protocol.json', protocol)
        # These files are placeholders for provenance tests, never executed.
        (self.root / 'learning.py').write_text('# synthetic fixture learning revision\n')
        (self.root / 'analyze_diagnostics.py').write_text('# synthetic fixture diagnostic revision\n')
        (self.root / 'audit_router_sensitivity.py').write_text('# synthetic fixture sensitivity revision\n')
        (self.root / 'plot_costs.py').write_text('# synthetic fixture cost-plot revision\n')
        write_json(self.root / 'analysis_protocol.json', {'synthetic_test_fixture': True})
        write_json(self.root / 'cache' / 'prepared.json', {'synthetic_test_fixture': True})
        self.headroom, self.adaptation, self.intervals, paired = [], [], [], {}
        setup = {'sample_calls': 16, 'llm_calls': 8, 'input_tokens': 80, 'output_tokens': 20}
        task_costs = [[{'search_calls': 2 if action == 4 else 1, 'llm_calls': int(action >= 2),
                       'input_tokens': 80 * int(action >= 2), 'output_tokens': 20 * int(action >= 2)}
                      for action in range(5)] for _ in range(2)]
        for name in self.environments:
            self.headroom.append({'environment': name, 'n_queries': '2', 'original': '0.525',
                'source_router': '0.675', 'target_fixed_oracle': '0.6', 'query_oracle': '0.9',
                'oracle_gap_vs_router': '0.225', 'oracle_gap_vs_target_fixed': '0.3',
                'original_recall_at_10': '0.75', 'source_router_recall_at_10': '0.875'})
            runs = {}
            for seed in seeds:
                for method in methods:
                    for budget in budgets:
                        active = method != 'source_router' and budget > 0
                        row = {'environment': name, 'seed': str(seed), 'method': method, 'budget': str(budget),
                               'ndcg': '0.675', 'recall_at_10': '0.875', 'changed_action_fraction': '0'}
                        row.update({f'onboarding_{key}': str(value if active else 0) for key, value in
                                    {'search_calls': 3, 'sample_calls': 16, 'llm_calls': 8, 'input_tokens': 80, 'output_tokens': 20}.items()})
                        row.update({f'mean_task_{key}': str(value) for key, value in
                                    {'search_calls': 1, 'sample_calls': 0, 'llm_calls': 0.5, 'input_tokens': 40, 'output_tokens': 10}.items()})
                        self.adaptation.append(row)
                        runs[f'{seed}/{method}/{budget}'] = {'actions': [1, 2], 'ndcg': [0.75, 0.6], 'recall_at_10': [1, 0.75],
                                                           'selected_probe_ids': ['synthetic-probe'] if active else []}
            for method in methods[1:]:
                for budget in budgets:
                    for comparator in ('source_router', 'random'):
                        self.intervals.append({'environment': name, 'method': method, 'budget': str(budget),
                            'comparator': comparator, 'mean': '0', 'lo': '0', 'hi': '0'})
            paired[name] = {'query_ids': ['synthetic-q0', 'synthetic-q1'], 'action_names': ['original', 'keywords', 'semantic', 'hyde', 'decomposed'],
                'action_scores': [[0.25, 0.75, 0.5, 0, 1], [0.8, 0.3, 0.6, 0.4, 0.2]],
                'action_recalls_at_10': [[0.5, 1, 0.5, 0, 1], [1, 0.25, 0.75, 0.5, 0.25]],
                'source_router_actions': [1, 2], 'runs': runs}
            write_json(self.root / 'cache' / f'{name}_outcomes.json', {'synthetic_test_fixture': name,
                'tasks': {'test': {'action_costs': task_costs}},
                'pools': {str(seed): {'setup_costs': setup, 'candidates': [{'id': 'synthetic-probe', 'cost': 3}]} for seed in seeds}})
        write_json(self.results / 'paired_outcomes.json', paired)
        write_csv(self.results / 'adaptation.csv', self.adaptation)
        self.cost_curves = []
        for row in self.adaptation:
            for workload in (1, 10, 50, 200):
                costs = {key: row[key] for key in ('environment', 'seed', 'method', 'budget', 'ndcg', 'recall_at_10')}
                costs['future_tasks'] = str(workload)
                costs.update({'total_' + key: str(float(row['onboarding_' + key]) + workload * float(row['mean_task_' + key]))
                    for key in ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens')})
                self.cost_curves.append(costs)
        write_csv(self.results / 'cost_curves.csv', self.cost_curves)
        write_csv(self.results / 'paired_intervals.csv', self.intervals)
        for name in ('headroom_intervals.csv', 'action_utilities.csv', 'probe_contrasts.csv'):
            write_csv(self.results / name, [{'synthetic_test_fixture': True}])
        self.fusion = [{'environment': name, 'n_queries': '2', 'ndcg': '0.9', 'recall_at_10': '0.9',
                        'mean_task_search_calls': '2', 'mean_task_output_tokens': '30'} for name in self.environments]
        write_csv(self.results / 'fusion_baseline.csv', self.fusion)
        (self.results / 'figures').mkdir()
        for name in ('headroom', 'adaptation', 'alignment'):
            (self.results / 'figures' / f'{name}.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        write_json(self.results / 'analysis_metadata.json', {
            'learning_sha256': digest(self.root / 'learning.py'), 'protocol_sha256': digest(self.root / 'protocol.json')})
        input_names = ['results/paired_outcomes.json', 'analysis_protocol.json', 'cache/prepared.json',
                       *[f'cache/{name}_outcomes.json' for name in self.environments]]
        self.provenance = {'code_sha256': digest(self.root / 'analyze_diagnostics.py'),
            'input_sha256': {name: digest(self.root / name) for name in input_names},
            'output_sha256': {name: digest(self.results / name) for name in
                ('paired_intervals.csv', 'headroom_intervals.csv', 'action_utilities.csv', 'probe_contrasts.csv',
                 'fusion_baseline.csv', 'figures/headroom.svg', 'figures/adaptation.svg', 'figures/alignment.svg')}}
        write_json(self.results / 'diagnostics_provenance.json', self.provenance)
        self.sensitivity = [{'environment': name, 'pair_budget': str(budget), 'n_queries': '2',
            'certified_unchangeable_fraction': '0.5', 'changeable_query_fraction_upper_bound': '0.5'}
            for name in self.environments for budget in budgets]
        write_csv(self.results / 'router_sensitivity.csv', self.sensitivity)
        write_json(self.results / 'router_sensitivity_queries.json', {'synthetic_test_fixture': True})
        write_json(self.results / 'models.json', {'synthetic_test_fixture': True})
        self.sensitivity_provenance = {'code_sha256': digest(self.root / 'audit_router_sensitivity.py'),
            'uses_unselected_probe_outcomes': False,
            'input_sha256': {name: digest(self.root / name) for name in
                ['results/models.json', 'results/paired_outcomes.json', 'results/analysis_metadata.json',
                 *[f'cache/{name}_outcomes.json' for name in self.environments]]},
            'output_sha256': {name: digest(self.results / name) for name in ('router_sensitivity.csv', 'router_sensitivity_queries.json')}}
        write_json(self.results / 'router_sensitivity_provenance.json', self.sensitivity_provenance)
        for category in ('search_calls', 'output_tokens'):
            (self.results / 'figures' / f'cost_{category}.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            (self.results / 'figures' / f'cost_{category}.png').write_bytes(b'synthetic image bytes for checksum tests')
        self.cost_provenance = {'code_sha256': digest(self.root / 'plot_costs.py'), 'validation_sha256': digest(Path(report.__file__)),
            'protocol_sha256': digest(self.root / 'protocol.json'), 'future_tasks': 50,
            'input_sha256': {name: digest(self.results / name) for name in ('cost_curves.csv', 'fusion_baseline.csv', 'adaptation.csv')},
            'output_sha256': {f'figures/cost_{category}.{suffix}': digest(self.results / 'figures' / f'cost_{category}.{suffix}')
                for category in ('search_calls', 'output_tokens') for suffix in ('png', 'svg')}}
        write_json(self.results / 'cost_plot_provenance.json', self.cost_provenance)
        self.root_patch = patch.object(report, 'ROOT', self.root)
        self.results_patch = patch.object(report, 'RESULTS', self.results)
        self.root_patch.start()
        self.results_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.results_patch.stop)

    def validate(self, headroom=None, adaptation=None, intervals=None):
        report.validate_results(self.headroom if headroom is None else headroom,
                                self.adaptation if adaptation is None else adaptation,
                                self.intervals if intervals is None else intervals)

    def test_complete_frozen_grid_and_matching_provenance_are_accepted(self):
        self.assertEqual(len(self.adaptation), 6 * 3 * 5 * 5)
        self.assertEqual(len(self.intervals), 6 * 4 * 5 * 2)
        self.assertEqual(len(self.cost_curves), 6 * 3 * 5 * 5 * 4)
        self.validate()

    def test_partial_duplicate_or_unexpected_adaptation_grid_is_rejected(self):
        unexpected = copy.deepcopy(self.adaptation)
        unexpected[0]['seed'] = '99'
        duplicate = [*self.adaptation[:-1], self.adaptation[0]]
        for label, rows in [('partial', self.adaptation[:-1]), ('duplicate', duplicate), ('unexpected', unexpected)]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'adaptation grid'):
                self.validate(adaptation=rows)

    def test_partial_duplicate_or_unexpected_interval_grid_is_rejected(self):
        unexpected = copy.deepcopy(self.intervals)
        unexpected[0]['comparator'] = 'target_oracle'
        duplicate = [*self.intervals[:-1], self.intervals[0]]
        for label, rows in [('partial', self.intervals[:-1]), ('duplicate', duplicate), ('unexpected', unexpected)]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'uncertainty grid'):
                self.validate(intervals=rows)

    def test_missing_or_duplicated_headroom_environment_is_rejected(self):
        for rows in (self.headroom[:-1], [*self.headroom[:-1], self.headroom[0]]):
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(ValueError, 'environment exactly once'):
                self.validate(headroom=rows)

    def test_missing_paired_environment_is_rejected(self):
        path = self.results / 'paired_outcomes.json'
        saved = json.loads(path.read_text())
        saved.pop(self.environments[0])
        write_json(path, saved)
        with self.assertRaisesRegex(ValueError, 'environment exactly once'):
            self.validate()

    def test_stale_adaptation_ndcg_and_recall_are_rejected(self):
        for metric in ('ndcg', 'recall_at_10'):
            rows = copy.deepcopy(self.adaptation)
            rows[-1][metric] = '0.2'
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, 'metric disagrees'):
                self.validate(adaptation=rows)

    def test_stale_changed_action_fraction_and_costs_are_rejected(self):
        for field in ('changed_action_fraction', *[f'{prefix}_{key}' for prefix in ('onboarding', 'mean_task')
                for key in ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens')]):
            rows = copy.deepcopy(self.adaptation)
            rows[-1][field] = '999'
            message = 'action-change fraction disagrees' if field == 'changed_action_fraction' else 'costs disagree'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.validate(adaptation=rows)

    def test_stale_headroom_scores_gaps_recalls_and_count_are_rejected(self):
        for field in ('n_queries', 'original', 'source_router', 'query_oracle', 'target_fixed_oracle',
                      'oracle_gap_vs_router', 'oracle_gap_vs_target_fixed', 'original_recall_at_10', 'source_router_recall_at_10'):
            rows = copy.deepcopy(self.headroom)
            rows[0][field] = '3' if field == 'n_queries' else '0.1'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Headroom disagrees'):
                self.validate(headroom=rows)

    def test_stale_code_protocol_input_and_diagnostic_output_are_rejected(self):
        names = ('learning.py', 'protocol.json', 'analyze_diagnostics.py', 'analysis_protocol.json',
                 'cache/alpha_bm25_outcomes.json', 'results/paired_intervals.csv', 'results/figures/headroom.svg',
                 'audit_router_sensitivity.py', 'results/models.json', 'results/router_sensitivity.csv',
                 'plot_costs.py', 'results/cost_curves.csv', 'results/adaptation.csv', 'results/figures/cost_search_calls.svg')
        for name in names:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            try:
                with self.subTest(file=name), self.assertRaisesRegex(ValueError, 'evidence drift'):
                    self.validate()
            finally:
                path.write_bytes(original)

    def test_missing_evidence_hash_is_rejected(self):
        for filename, metadata in [('diagnostics_provenance.json', self.provenance),
                                   ('router_sensitivity_provenance.json', self.sensitivity_provenance),
                                   ('cost_plot_provenance.json', self.cost_provenance)]:
            for section in ('input_sha256', 'output_sha256'):
                changed = copy.deepcopy(metadata)
                changed[section].pop(next(iter(changed[section])))
                write_json(self.results / filename, changed)
                try:
                    with self.subTest(file=filename, section=section), self.assertRaisesRegex(ValueError, 'evidence provenance'):
                        self.validate()
                finally:
                    write_json(self.results / filename, metadata)

    def save_cost_fixture(self, rows):
        write_csv(self.results / 'cost_curves.csv', rows)
        provenance = copy.deepcopy(self.cost_provenance)
        provenance['input_sha256']['cost_curves.csv'] = digest(self.results / 'cost_curves.csv')
        write_json(self.results / 'cost_plot_provenance.json', provenance)

    def test_partial_duplicate_or_unexpected_cost_grid_is_rejected(self):
        unexpected = copy.deepcopy(self.cost_curves)
        unexpected[0]['future_tasks'] = '99'
        duplicate = [*self.cost_curves[:-1], self.cost_curves[0]]
        for label, rows in [('partial', self.cost_curves[:-1]), ('duplicate', duplicate), ('unexpected', unexpected)]:
            self.save_cost_fixture(rows)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'cost-curve grid'):
                self.validate()

    def test_cost_curve_quality_and_all_resource_totals_must_match_the_run(self):
        for field in ('ndcg', 'recall_at_10', *['total_' + key for key in
                      ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens')]):
            rows = copy.deepcopy(self.cost_curves)
            rows[-1][field] = '0.123'
            self.save_cost_fixture(rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Cost-curve .* disagrees'):
                self.validate()

    def test_cost_plot_workload_and_validation_revision_are_bound(self):
        for field, value, message in [('future_tasks', 200, 'evidence provenance'),
                                      ('validation_sha256', 'wrong-revision', 'evidence drift')]:
            provenance = copy.deepcopy(self.cost_provenance)
            provenance[field] = value
            write_json(self.results / 'cost_plot_provenance.json', provenance)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.validate()

    def test_standalone_cost_validation_requires_an_invariant_source_router(self):
        protocol = json.loads((self.root / 'protocol.json').read_text())
        for field in ('onboarding_search_calls', 'mean_task_output_tokens', 'ndcg'):
            rows = copy.deepcopy(self.adaptation)
            rows[1][field] = '0.123'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Source-router quality or costs vary'):
                report.validate_cost_curves(self.cost_curves, rows, protocol)

    def test_cost_plot_embeds_fifty_task_totals_and_records_complete_provenance(self):
        import plot_costs
        from matplotlib.figure import Figure

        (self.root / 'plot_costs.py').write_bytes(Path(plot_costs.__file__).read_bytes())
        displayed = {}
        def capture(figure, path, **_):
            path = Path(path)
            axes = figure.axes[0]
            displayed[path.stem] = {'curve_x': axes.lines[0].get_xdata().tolist(),
                'curve_y': axes.lines[0].get_ydata().tolist(),
                'baseline': axes.collections[0].get_offsets().tolist(), 'fusion': axes.collections[1].get_offsets().tolist()}
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>' if path.suffix == '.svg' else 'synthetic PNG fixture')
        with patch.object(plot_costs, 'ROOT', self.root), patch.object(plot_costs, 'RESULTS', self.results), patch.object(Figure, 'savefig', capture):
            plot_costs.main()
        self.assertEqual(displayed['cost_search_calls']['curve_x'], [50, 53, 53, 53, 53])
        self.assertEqual(displayed['cost_output_tokens']['curve_x'], [500, 520, 520, 520, 520])
        self.assertEqual(displayed['cost_search_calls']['baseline'], [[50, 0.675]])
        self.assertEqual(displayed['cost_search_calls']['fusion'], [[100, 0.9]])
        self.assertEqual(displayed['cost_output_tokens']['fusion'], [[1500, 0.9]])
        provenance = json.loads((self.results / 'cost_plot_provenance.json').read_text())
        self.assertEqual(len(provenance['output_sha256']), 4)
        self.assertEqual(set(provenance['input_sha256']), {'cost_curves.csv', 'adaptation.csv', 'fusion_baseline.csv'})
        self.validate()

    def test_standalone_cost_plot_rejects_duplicate_or_invalid_fusion_rows(self):
        import plot_costs

        duplicate = [*self.fusion[:-1], self.fusion[0]]
        fixtures = [('duplicate', duplicate)]
        for field, value in [('n_queries', '3'), ('ndcg', 'nan'), ('mean_task_output_tokens', '-1')]:
            rows = copy.deepcopy(self.fusion)
            rows[0][field] = value
            fixtures.append((field, rows))
        for label, rows in fixtures:
            write_csv(self.results / 'fusion_baseline.csv', rows)
            with self.subTest(label=label), patch.object(plot_costs, 'ROOT', self.root), patch.object(plot_costs, 'RESULTS', self.results), self.assertRaisesRegex(ValueError, '[Ff]usion'):
                plot_costs.main()

    def save_sensitivity_fixture(self, rows):
        write_csv(self.results / 'router_sensitivity.csv', rows)
        provenance = copy.deepcopy(self.sensitivity_provenance)
        provenance['output_sha256']['router_sensitivity.csv'] = digest(self.results / 'router_sensitivity.csv')
        write_json(self.results / 'router_sensitivity_provenance.json', provenance)

    def test_partial_duplicate_or_unexpected_sensitivity_grid_is_rejected(self):
        unexpected = copy.deepcopy(self.sensitivity)
        unexpected[0]['pair_budget'] = '99'
        duplicate = [*self.sensitivity[:-1], self.sensitivity[0]]
        for label, rows in [('partial', self.sensitivity[:-1]), ('duplicate', duplicate), ('unexpected', unexpected)]:
            self.save_sensitivity_fixture(rows)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'sensitivity grid'):
                self.validate()

    def test_sensitivity_query_count_and_complementary_fractions_are_checked(self):
        for field, value in [('n_queries', '3'), ('certified_unchangeable_fraction', 'nan'),
                             ('changeable_query_fraction_upper_bound', '1.5'), ('certified_unchangeable_fraction', '0.1')]:
            rows = copy.deepcopy(self.sensitivity)
            rows[0][field] = value
            self.save_sensitivity_fixture(rows)
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'Sensitivity cohort or fraction'):
                self.validate()


class EmbeddedFigureTests(unittest.TestCase):
    def test_independent_svg_ids_and_references_are_namespaced_and_caption_escaped(self):
        svg = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<defs><path id="glyph" d="M0 0L1 1"/><clipPath id="clip"><rect width="1" height="1"/></clipPath></defs>
<g id="axes" clip-path="url(#clip)"><use xlink:href="#glyph"/><use href="#glyph"/></g>
</svg>'''
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / 'figures').mkdir()
            for name in ('headroom', 'adaptation'):
                (results / 'figures' / f'{name}.svg').write_text(svg)
            with patch.object(report, 'RESULTS', results):
                fragments = [report.figure(name, 'Quality < 1 & "caption"') for name in ('headroom', 'adaptation')]
        all_ids = []
        for name, fragment in zip(('headroom', 'adaptation'), fragments, strict=True):
            self.assertNotIn('<?xml', fragment)
            root = ET.fromstring(fragment)
            image = root.find('{http://www.w3.org/2000/svg}svg')
            self.assertEqual(image.attrib['role'], 'img')
            self.assertEqual(image.attrib['aria-label'], 'Quality < 1 & "caption"')
            self.assertEqual(root.find('figcaption').text, 'Quality < 1 & "caption"')
            ids = {element.attrib['id'] for element in image.iter() if 'id' in element.attrib}
            self.assertEqual(ids, {f'{name}_{suffix}' for suffix in ('glyph', 'clip', 'axes')})
            references = set(re.findall(r'(?:href="#|url\(#)([^"\)]+)', fragment))
            self.assertEqual(references, {f'{name}_glyph', f'{name}_clip'})
            self.assertTrue(references <= ids)
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == '__main__':
    unittest.main()
