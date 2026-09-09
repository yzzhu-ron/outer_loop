import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pilot import CachedSearch, Profile, onboard, rrf, select_ids


class FakeSearch:
    def __init__(self):
        self.queries = []

    def batch(self, queries):
        self.queries.extend(queries)
        return [['known'] for _ in queries]


class ContractTests(unittest.TestCase):
    def test_onboarding_does_not_accept_tasks_or_labels(self):
        self.assertEqual(list(inspect.signature(onboard).parameters),
                         ['sampled_documents', 'search', 'family', 'budget'])

    def test_budget_charges_sampling_and_both_searches(self):
        docs = [{'_id': 'known', 'title': 'A technical document title', 'text': 'body'}] * 10
        search = FakeSearch()
        result = onboard(docs, search, 'exact_title', 16)
        self.assertEqual(result['operations'], 15)
        self.assertEqual(result['sampled_documents'], 5)
        self.assertEqual(len(search.queries), 10)
        self.assertEqual(np.asarray(result['profile']['count']).sum(), 5)

    def test_invalid_probe_still_charges_sampling(self):
        docs = [{'_id': 'private-doc-id', 'title': '', 'text': 'secret answer'}] * 10
        search = FakeSearch()
        result = onboard(docs, search, 'exact_title', 5)
        self.assertLessEqual(result['operations'], 5)
        self.assertGreater(result['operations'], 0)
        self.assertEqual(result['operations'], result['sampled_documents'])
        self.assertEqual(search.queries, [])
        self.assertNotIn('secret', json.dumps(result))
        self.assertNotIn('private-doc-id', json.dumps(result))

    def test_zero_budget_does_not_inspect_document(self):
        class NeverInspect:
            def __iter__(self):
                raise AssertionError('zero budget inspected documents')
        # A sequence read of a document is not required at zero budget.
        result = onboard(NeverInspect(), FakeSearch(), 'exact_title', 0)
        self.assertEqual(result['operations'], 0)
        self.assertEqual(result['sampled_documents'], 0)

    def test_profile_is_numeric_bounded_and_roundtrips(self):
        profile = Profile()
        for _ in range(1000):
            profile.update(1, 2, .5)
        self.assertEqual(profile.count.shape, (4, 3))
        saved = profile.to_dict()
        self.assertEqual(set(saved), {'count', 'sum', 'sum_sq'})
        self.assertEqual(Profile.from_dict(saved).to_dict(), saved)
        self.assertEqual(Profile.from_dict(saved).features(2, 0), [0.0, 0.0])

    def test_id_selection_independent_of_input_order(self):
        self.assertEqual(select_ids(['c', 'a', 'b'], 2, 'fixed'),
                         select_ids(['b', 'c', 'a'], 2, 'fixed'))

    def test_rrf_can_promote_agreed_evidence(self):
        self.assertEqual(rrf([['a', 'b'], ['c', 'b']], [0, 1])[0], 'b')

    def test_cache_charges_reused_observations_and_invalidates_identity(self):
        class Backend:
            calls = 0
            def batch_search(self, queries, k):
                self.calls += len(queries)
                return [['a'] for q in queries]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ranks.json'
            backend = Backend()
            cache = CachedSearch(backend, path, 'v1')
            cache.batch(['q', 'q'])
            cache.batch(['q'])
            self.assertEqual(cache.calls, 3)
            self.assertEqual(backend.calls, 1)
            cache.save()
            self.assertFalse(CachedSearch(backend, path, 'v2').ranks)
            self.assertTrue(CachedSearch(backend, path, 'v1').ranks)


if __name__ == '__main__':
    unittest.main()
