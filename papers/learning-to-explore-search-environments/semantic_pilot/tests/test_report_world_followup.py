"""Corruption checks for the report using a complete, temporary synthetic run."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import validate_cost_curves, validate_cost_rows
import report_world_followup as report
import world_model_followup as world
from test_world_model_followup import records_fixture


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class FollowupReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results'
        self.followup = self.results / 'world_model_followup'
        self.results.mkdir()
        (self.root / 'cache').mkdir()
        records, self.original = records_fixture()
        self.names = sorted(self.original)
        self.first = self.names[0]
        protocol = {
            'datasets': {name: {} for name in ('alpha', 'beta', 'gamma')},
            'probes': {'seeds': list(world.SEEDS), 'pair_budgets': list(world.BUDGETS)},
            'query_selection': {'test_counts': {name: 3 for name in ('alpha', 'beta', 'gamma')}},
        }
        write_json(self.root / 'protocol.json', protocol)
        write_json(self.results / 'paired_outcomes.json', self.original)
        for name in ('analysis_metadata.json', 'diagnostics_provenance.json'):
            write_json(self.results / name, {'synthetic_fixture': True})
        # Copies are checksum fixtures. Evaluation still executes the frozen
        # module against synthetic records, never against real experiment data.
        for name in ('world_model_followup.py', 'learning.py', 'world_model_protocol.json'):
            (self.root / name).write_bytes((world.ROOT / name).read_bytes())
        for record in records:
            write_json(self.root / 'cache' / f"{record['name']}_outcomes.json", record)
        input_names = [
            'results/analysis_metadata.json', 'results/paired_outcomes.json',
            'results/diagnostics_provenance.json', 'protocol.json', 'world_model_protocol.json',
            *[f'cache/{name}_outcomes.json' for name in self.names],
        ]
        world.evaluate_records(
            records, self.original, self.followup,
            protocol=json.loads(world.PROTOCOL_PATH.read_text()),
            input_provenance={name: digest(self.root / name) for name in input_names},
        )
        self.pristine = {path.relative_to(self.root): path.read_bytes()
                         for path in self.root.rglob('*') if path.is_file()}

    def restore(self):
        for name, content in self.pristine.items():
            (self.root / name).write_bytes(content)

    def validate(self):
        with patch.multiple(report, ROOT=self.root, RESULTS=self.results, FOLLOWUP=self.followup):
            return report.load_checked(validate_cost_rows, validate_cost_curves)

    def change_metadata(self, change):
        path = self.followup / 'analysis_metadata.json'
        metadata = json.loads(path.read_text())
        change(metadata)
        write_json(path, metadata)

    def change_output(self, name, change):
        """Rehash deliberate corruption to exercise checks beyond byte drift."""
        path = self.followup / name
        if path.suffix == '.json':
            value = json.loads(path.read_text())
            change(value)
            write_json(path, value)
        else:
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                fields, value = reader.fieldnames, list(reader)
            change(value)
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(value)
        self.change_metadata(lambda metadata: metadata['output_sha256'].__setitem__(name, digest(path)))

    def test_complete_synthetic_evidence_is_accepted(self):
        names, adaptation, intervals, baselines = self.validate()
        self.assertEqual(names, self.names)
        self.assertEqual(len(adaptation), 360)
        self.assertEqual(len(intervals), 360)
        self.assertEqual(len(baselines), 18)

    def test_exact_input_and_output_provenance_sets_are_required(self):
        for field, filename, message in (
            ('input_provenance', 'cache/alpha_bm25_outcomes.json', 'input provenance'),
            ('output_sha256', 'models.json', 'output provenance'),
        ):
            for mutation in ('missing', 'unexpected'):
                with self.subTest(field=field, mutation=mutation):
                    self.restore()
                    def change(metadata):
                        if mutation == 'missing':
                            metadata[field].pop(filename)
                        else:
                            metadata[field]['unexpected.json'] = '0' * 64
                    self.change_metadata(change)
                    with self.assertRaisesRegex(ValueError, message):
                        self.validate()

    def test_code_input_and_output_byte_drift_is_rejected(self):
        for name in (
            'world_model_followup.py', 'learning.py', 'protocol.json', 'world_model_protocol.json',
            'cache/alpha_bm25_outcomes.json', 'results/paired_outcomes.json',
            'results/analysis_metadata.json', 'results/diagnostics_provenance.json',
            'results/world_model_followup/models.json',
            'results/world_model_followup/posterior_traces.json',
            'results/world_model_followup/adaptation.csv',
        ):
            with self.subTest(name=name):
                self.restore()
                path = self.root / name
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'Follow-up evidence drift'):
                    self.validate()

    def test_protocol_metadata_must_match_frozen_file(self):
        for field, value in (
            ('protocol', {'wrong_protocol': True}),
            ('protocol_canonical_sha256', '0' * 64),
            ('protocol_file_sha256', '0' * 64),
        ):
            with self.subTest(field=field):
                self.restore()
                self.change_metadata(lambda metadata: metadata.__setitem__(field, value))
                with self.assertRaisesRegex(ValueError, 'protocol metadata mismatch'):
                    self.validate()

    def test_missing_and_duplicate_grids_are_rejected(self):
        cases = (
            ('paired_outcomes.json', lambda data: data.pop(self.first), 'environments'),
            ('paired_outcomes.json', lambda data: data[self.first]['runs'].pop('11/random/0'), 'paired run grid'),
            ('baselines.csv', lambda rows: rows.pop(), 'baseline grid'),
            ('baselines.csv', lambda rows: rows.__setitem__(-1, rows[0].copy()), 'baseline grid'),
            ('adaptation.csv', lambda rows: rows.pop(), 'run grid'),
            ('adaptation.csv', lambda rows: rows.__setitem__(-1, rows[0].copy()), 'run grid'),
            ('paired_intervals.csv', lambda rows: rows.pop(), 'interval grid'),
            ('paired_intervals.csv', lambda rows: rows.__setitem__(-1, rows[0].copy()), 'interval grid'),
        )
        for index, (filename, change, message) in enumerate(cases):
            with self.subTest(case=index, file=filename):
                self.restore()
                self.change_output(filename, change)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_actions_require_in_range_integers_and_complete_cohort(self):
        for source in ('baseline', 'run'):
            for invalid in (-1, True, 1.0, 5, '1', None, 'short_list'):
                with self.subTest(source=source, invalid=invalid):
                    self.restore()
                    def change(data):
                        environment = data[self.first]
                        item = (environment['baselines']['zero_budget'] if source == 'baseline'
                                else environment['runs']['11/random/4'])
                        if invalid == 'short_list':
                            item['actions'].pop()
                        else:
                            item['actions'][0] = invalid
                    self.change_output('paired_outcomes.json', change)
                    with self.assertRaisesRegex(ValueError, 'Invalid follow-up actions'):
                        self.validate()

    def test_query_cohort_and_reference_baselines_are_frozen(self):
        cases = (
            (lambda data: data[self.first]['query_ids'].reverse(), 'query cohort or baselines changed'),
            (lambda data: data[self.first]['baselines'].pop('original_query'), 'query cohort or baselines changed'),
            (lambda data: data[self.first]['baselines']['original_frozen_router']['actions'].__setitem__(0, 1), 'reference baseline changed'),
            (lambda data: data[self.first]['baselines']['original_query']['actions'].__setitem__(0, 1), 'reference baseline changed'),
        )
        for index, (change, message) in enumerate(cases):
            with self.subTest(case=index):
                self.restore()
                self.change_output('paired_outcomes.json', change)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_baseline_cohort_and_quality_are_reconstructed(self):
        self.change_output('baselines.csv', lambda rows: rows[0].__setitem__('n_queries', '2'))
        with self.assertRaisesRegex(ValueError, 'baseline cohort mismatch'):
            self.validate()
        for metric in ('ndcg', 'recall_at_10'):
            for source in ('paired', 'csv'):
                with self.subTest(metric=metric, source=source):
                    self.restore()
                    if source == 'paired':
                        self.change_output('paired_outcomes.json', lambda data:
                            data[self.first]['baselines']['zero_budget'][metric].__setitem__(0, 0.123))
                    else:
                        self.change_output('baselines.csv', lambda rows: rows[0].__setitem__(metric, '0.123'))
                    with self.assertRaisesRegex(ValueError, 'baseline quality mismatch'):
                        self.validate()

    def test_all_baseline_serving_costs_are_reconstructed(self):
        for resource in ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens'):
            with self.subTest(resource=resource):
                self.restore()
                key = 'mean_task_' + resource
                self.change_output('baselines.csv', lambda rows: rows[0].__setitem__(key, str(float(rows[0][key]) + 1)))
                with self.assertRaisesRegex(ValueError, 'baseline cost mismatch'):
                    self.validate()

    def test_zero_budget_cannot_observe_probes_or_depart_from_own_prior(self):
        for mutation in ('observed_probe', 'changed_action'):
            with self.subTest(mutation=mutation):
                self.restore()
                replacement = {}
                def change(data):
                    run = data[self.first]['runs']['11/random/0']
                    if mutation == 'observed_probe':
                        run['selected_probe_ids'].append('p0')
                    else:
                        run['actions'][0] = (run['actions'][0] + 1) % 5
                        for metric, source in (('ndcg', 'action_scores'), ('recall_at_10', 'action_recalls_at_10')):
                            run[metric] = [values[action] for values, action in
                                           zip(self.original[self.first][source], run['actions'], strict=True)]
                            replacement[metric] = str(sum(run[metric]) / len(run[metric]))
                self.change_output('paired_outcomes.json', change)
                if replacement:
                    def update_quality(rows):
                        row = next(row for row in rows if (row['environment'], row['seed'], row['method'], row['budget'])
                                   == (self.first, '11', 'random', '0'))
                        row.update(replacement)
                    self.change_output('adaptation.csv', update_quality)
                with self.assertRaisesRegex(ValueError, 'budget zero differs from own prior'):
                    self.validate()

    def test_run_quality_and_action_change_fraction_are_reconstructed(self):
        for metric in ('ndcg', 'recall_at_10'):
            for source in ('paired', 'csv'):
                with self.subTest(metric=metric, source=source):
                    self.restore()
                    if source == 'paired':
                        self.change_output('paired_outcomes.json', lambda data:
                            data[self.first]['runs']['11/random/4'][metric].__setitem__(0, 0.123))
                    else:
                        self.change_output('adaptation.csv', lambda rows: rows[0].__setitem__(metric, '0.123'))
                    with self.assertRaisesRegex(ValueError, 'Follow-up quality mismatch'):
                        self.validate()
        self.restore()
        self.change_output('adaptation.csv', lambda rows: rows[0].__setitem__('changed_action_fraction_vs_zero_budget', '0.123'))
        with self.assertRaisesRegex(ValueError, 'action-change mismatch'):
            self.validate()

    def test_interval_means_and_finite_ordered_endpoints_are_required(self):
        for values in (
            {'mean': '0.123'}, {'mean': 'nan'}, {'lo': 'nan'}, {'hi': 'nan'},
            {'lo': '-inf'}, {'hi': 'inf'}, {'lo': '0.2', 'hi': '-0.2'},
            {'lo': '-1.01'}, {'hi': '1.01'},
        ):
            with self.subTest(values=values):
                self.restore()
                self.change_output('paired_intervals.csv', lambda rows: rows[0].update(values))
                with self.assertRaisesRegex(ValueError, 'interval contrast mismatch'):
                    self.validate()

    def test_percentile_endpoints_need_not_contain_the_point_estimate(self):
        # A percentile interval need not contain the sample statistic. This
        # checks the validator's mathematical contract, not bootstrap coverage.
        self.change_output('paired_intervals.csv', lambda rows: rows[0].update(lo='0.8', hi='0.9'))
        self.validate()

    def test_selected_probe_and_cost_ledger_corruption_is_rejected(self):
        cases = (
            ('paired_outcomes.json', lambda data: data[self.first]['runs']['11/random/4']['selected_probe_ids'].append('unknown'), 'Invalid selected probes'),
            ('paired_outcomes.json', lambda data: data[self.first]['runs']['11/random/4']['selected_probe_ids'].__setitem__(1, data[self.first]['runs']['11/random/4']['selected_probe_ids'][0]), 'Invalid selected probes'),
            ('adaptation.csv', lambda rows: rows[1].__setitem__('onboarding_search_calls', '999'), 'Report costs disagree'),
            ('adaptation.csv', lambda rows: rows[1].__setitem__('mean_task_output_tokens', '999'), 'Report costs disagree'),
            ('cost_curves.csv', lambda rows: rows.pop(), 'cost-curve grid'),
            ('cost_curves.csv', lambda rows: rows.__setitem__(-1, rows[0].copy()), 'cost-curve grid'),
            ('cost_curves.csv', lambda rows: rows[0].__setitem__('total_search_calls', '999'), 'Cost-curve total disagrees'),
            ('cost_curves.csv', lambda rows: rows[0].__setitem__('total_search_calls', 'nan'), 'Cost-curve total disagrees'),
            ('cost_curves.csv', lambda rows: rows[0].__setitem__('ndcg', '0.123'), 'Cost-curve quality disagrees'),
        )
        for index, (filename, change, message) in enumerate(cases):
            with self.subTest(case=index, file=filename):
                self.restore()
                self.change_output(filename, change)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()


if __name__ == '__main__':
    unittest.main()
