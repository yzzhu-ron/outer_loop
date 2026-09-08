"""
proposer.py
Harness proposer: uses the frozen LLM (Claude) to propose a new harness
configuration given execution traces and the current archive.
Also implements an unconditioned (random) proposer for gamma measurement.
"""

import json
import random
import os
from typing import Optional

from outer_loop.archive import ParetoArchive
from shopgym.harness import (
    ACTION_VOCABS,
    CTX_WINDOWS,
    OBS_MODALITIES,
    RETRY_POLICIES,
    SCAFFOLDS,
    HarnessConfig,
    random_harness,
)


SCHEMA_DESCRIPTION = """
Harness axes and allowed values:
- obs_modality: screenshot | ax_tree | dom_text | hybrid_ax_ss | hybrid_dom_ss
- action_vocab: low_level | high_level | mixed
- ctx_window:   last_1 | last_3 | last_5 | full_summary
- scaffold:     none | step_counter | error_overlay | task_decomp | full
- retry_policy: none | once | backtrack
"""

PROPOSER_SYSTEM_PROMPT = f"""You are a harness engineer for web automation agents.
Your task is to propose a harness configuration that improves task completion rate on e-commerce tasks.

{SCHEMA_DESCRIPTION}

Key definitions:
- obs_modality: how the browser state is encoded for the agent
  - screenshot: raw image only (good for visual forms and payment pages)
  - ax_tree: accessibility tree only (good for structured search results)
  - hybrid_ax_ss: both ax_tree and screenshot (balanced, good for multi-step tasks)
  - dom_text: page text only (fastest, lowest quality)
  - hybrid_dom_ss: dom_text + screenshot
- action_vocab: what actions the agent can output
  - high_level: semantic actions (CLICK by label, TYPE by field name) — easier to parse
  - low_level: pixel coordinates — more precise but brittle
  - mixed: try high_level first, fallback to coordinates
- ctx_window: how many prior steps to include in the agent's context
  - full_summary: use an LLM to summarize prior steps — best for long multi-step tasks
- scaffold: extra information injected into observations
  - full: includes step counter, error overlays, and task decomposition — most expensive
- retry_policy: what to do when an action fails
  - backtrack: go to previous page — critical for checkout flows

Output ONLY valid JSON with the five axes. No prose, no explanation.
"""


def format_traces_for_proposer(traces: list, max_traces: int = 10) -> str:
    """Format episode traces into a compact string for the proposer prompt."""
    lines = []
    for i, ep in enumerate(traces[-max_traces:]):
        if hasattr(ep, "failure_mode"):
            # EpisodeResult
            lines.append(f"Episode {i+1}: success={ep.success}, "
                         f"stages_completed={ep.stages_completed}, "
                         f"failure_mode={ep.failure_mode}")
            for t in ep.trace:
                lines.append(f"  Stage {t.get('stage', '?')}: "
                             f"p_success={t.get('p_success', '?'):.2f}, "
                             f"outcome={'✓' if t.get('success') else '✗'}")
        else:
            lines.append(str(ep))
    return "\n".join(lines)


class LLMHarnessProposer:
    """
    Trace-conditioned harness proposer using the frozen LLM backbone.
    """

    def __init__(self, model: str = "claude-sonnet-4-6",
                 temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

    def propose(self, traces: list, parent_harness: HarnessConfig,
                archive: Optional[ParetoArchive] = None) -> HarnessConfig:
        """
        Propose a new harness configuration conditioned on execution traces.

        Args:
            traces: list of EpisodeResult from running parent_harness
            parent_harness: the harness that generated the traces
            archive: current Pareto archive (to avoid re-proposing existing configs)

        Returns:
            New HarnessConfig
        """
        user_msg = self._build_user_message(traces, parent_harness, archive)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=256,
                system=PROPOSER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=self.temperature,
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            if "{" in text:
                text = text[text.index("{"):text.rindex("}")+1]
            proposal = json.loads(text)
            return HarnessConfig.from_dict(proposal)
        except Exception as e:
            print(f"Proposer error: {e}. Falling back to random proposal.")
            return random_harness()

    def _build_user_message(self, traces: list, parent_harness: HarnessConfig,
                             archive: Optional[ParetoArchive]) -> str:
        msg = f"Current harness:\n{parent_harness.to_json()}\n\n"
        msg += f"Failure traces ({len(traces)} recent episodes):\n"
        msg += format_traces_for_proposer(traces) + "\n\n"
        if archive and archive.entries:
            msg += "Archive (do not reproduce these exactly):\n"
            for e in archive.entries[:5]:
                msg += f"  {e.harness.to_dict()} → rate={e.completion_rate:.3f}\n"
            msg += "\n"
        msg += "Propose a new harness JSON that addresses the observed failure modes:"
        return msg


class UnconditionedProposer:
    """
    Baseline proposer that generates uniformly random harness configurations,
    ignoring traces.  Used to measure p_0 for gamma estimation.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = random.Random(rng_seed)

    def propose(self, traces: list, parent_harness: HarnessConfig,
                archive: Optional[ParetoArchive] = None) -> HarnessConfig:
        return random_harness(self.rng)


class MockLLMProposer:
    """
    A deterministic mock proposer that simulates LLM behavior using
    trace analysis heuristics.  Used when no API key is available.

    Implements the same interface as LLMHarnessProposer.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = random.Random(rng_seed)
        # Track how many proposals have been made (to simulate improving proposals)
        self._proposal_count = 0

    def propose(self, traces: list, parent_harness: HarnessConfig,
                archive: Optional[ParetoArchive] = None) -> HarnessConfig:
        self._proposal_count += 1

        # Analyze failure modes in traces
        failure_modes = [ep.failure_mode for ep in traces if hasattr(ep, "failure_mode")
                         and ep.failure_mode is not None]

        # Heuristic rules that simulate trace-conditioned proposals
        new_config = parent_harness.to_dict().copy()

        if "visual_form_missing" in failure_modes:
            # The ax_tree harness fails on visual forms → add screenshot
            if new_config["obs_modality"] in ("ax_tree", "dom_text"):
                new_config["obs_modality"] = self.rng.choice(["hybrid_ax_ss", "screenshot"])

        if "context_lost" in failure_modes:
            # last_1 context is too short → expand
            if new_config["ctx_window"] in ("last_1", "last_3"):
                new_config["ctx_window"] = self.rng.choice(["last_5", "full_summary"])

        if "action_failed" in failure_modes:
            # Actions failing → try backtrack or mixed vocab
            new_config["retry_policy"] = "backtrack"
            if new_config["action_vocab"] == "low_level":
                new_config["action_vocab"] = "mixed"

        # Add scaffold as proposals mature
        if self._proposal_count > 5 and new_config["scaffold"] == "none":
            new_config["scaffold"] = self.rng.choice(["step_counter", "error_overlay",
                                                        "task_decomp", "full"])

        # Small random perturbation to ensure variety
        if self.rng.random() < 0.3:
            axis = self.rng.choice(["obs_modality", "action_vocab",
                                     "ctx_window", "scaffold", "retry_policy"])
            choices = {
                "obs_modality": OBS_MODALITIES,
                "action_vocab": ACTION_VOCABS,
                "ctx_window":   CTX_WINDOWS,
                "scaffold":     SCAFFOLDS,
                "retry_policy": RETRY_POLICIES,
            }[axis]
            new_config[axis] = self.rng.choice(choices)

        try:
            return HarnessConfig.from_dict(new_config)
        except Exception:
            return random_harness(self.rng)
