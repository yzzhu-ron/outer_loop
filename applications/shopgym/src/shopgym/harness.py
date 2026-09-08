"""ShopGym harness artifact and its five-axis search space."""

from dataclasses import asdict, dataclass
import itertools
import json

OBS_MODALITIES  = ["screenshot", "ax_tree", "dom_text", "hybrid_ax_ss", "hybrid_dom_ss"]
ACTION_VOCABS   = ["low_level", "high_level", "mixed"]
CTX_WINDOWS     = ["last_1", "last_3", "last_5", "full_summary"]
SCAFFOLDS       = ["none", "step_counter", "error_overlay", "task_decomp", "full"]
RETRY_POLICIES  = ["none", "once", "backtrack"]


@dataclass(frozen=True)
class HarnessConfig:
    obs_modality: str = "screenshot"
    action_vocab: str = "high_level"
    ctx_window:   str = "last_3"
    scaffold:     str = "none"
    retry_policy: str = "once"

    def __post_init__(self):
        allowed = {
            "obs_modality": OBS_MODALITIES,
            "action_vocab": ACTION_VOCABS,
            "ctx_window": CTX_WINDOWS,
            "scaffold": SCAFFOLDS,
            "retry_policy": RETRY_POLICIES,
        }
        for field_name, choices in allowed.items():
            value = getattr(self, field_name)
            if value not in choices:
                raise ValueError(f"invalid {field_name}: {value}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "HarnessConfig":
        return cls(**{k: d[k] for k in ["obs_modality", "action_vocab",
                                         "ctx_window", "scaffold", "retry_policy"]})

    @classmethod
    def from_json(cls, s: str) -> "HarnessConfig":
        return cls.from_dict(json.loads(s))

def default_harness() -> HarnessConfig:
    return HarnessConfig(
        obs_modality="screenshot",
        action_vocab="high_level",
        ctx_window="last_3",
        scaffold="none",
        retry_policy="once",
    )


def best_checkout_harness() -> HarnessConfig:
    """Best harness found by GROL for checkout tasks."""
    return HarnessConfig(
        obs_modality="hybrid_ax_ss",
        action_vocab="mixed",
        ctx_window="full_summary",
        scaffold="full",
        retry_policy="backtrack",
    )


def enumerate_all_harnesses() -> list:
    """Returns all 900 harness configurations."""
    configs = []
    for combo in itertools.product(OBS_MODALITIES, ACTION_VOCABS,
                                   CTX_WINDOWS, SCAFFOLDS, RETRY_POLICIES):
        configs.append(HarnessConfig(*combo))
    return configs


def random_harness(rng=None) -> HarnessConfig:
    import random
    r = rng or random
    return HarnessConfig(
        obs_modality=r.choice(OBS_MODALITIES),
        action_vocab=r.choice(ACTION_VOCABS),
        ctx_window=r.choice(CTX_WINDOWS),
        scaffold=r.choice(SCAFFOLDS),
        retry_policy=r.choice(RETRY_POLICIES),
    )
