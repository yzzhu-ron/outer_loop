import unittest
from dataclasses import dataclass

from outer_loop import GatedRatchet, ParetoArchive, wilson_ci


@dataclass(frozen=True)
class Artifact:
    name: str


@dataclass
class Result:
    artifact: Artifact
    score: float
    interval: object
    objectives: tuple[float, ...]


class CoreTests(unittest.TestCase):
    def test_ratchet_requires_separated_intervals(self):
        baseline = Result(Artifact("base"), 0.2, wilson_ci(20, 100), (0.2,))
        candidate = Result(Artifact("candidate"), 0.8, wilson_ci(80, 100), (0.8,))
        ratchet = GatedRatchet()
        ratchet.set_incumbent(baseline)
        self.assertTrue(ratchet.accept(candidate))
        self.assertIs(ratchet.incumbent, candidate)

    def test_archive_removes_dominated_entries(self):
        weak = Result(Artifact("weak"), 0.4, wilson_ci(4, 10), (0.4, 0.5))
        strong = Result(Artifact("strong"), 0.6, wilson_ci(6, 10), (0.6, 0.7))
        archive = ParetoArchive()
        self.assertTrue(archive.try_add(weak))
        self.assertTrue(archive.try_add(strong))
        self.assertEqual([strong], archive.entries)


if __name__ == "__main__":
    unittest.main()
