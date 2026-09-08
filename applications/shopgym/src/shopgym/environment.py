"""ShopGym mock and Playwright-backed experiment environments."""

import random
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from shopgym.harness import HarnessConfig


# ── Task stage definitions ───────────────────────────────────────────────────

TASK_STAGES = {
    "product_search":              ["search"],
    "add_to_cart":                 ["search", "add_to_cart"],
    "multi_item_cart":             ["search", "add_item1", "add_item2"],
    "checkout_single_item":        ["search", "add_to_cart", "begin_checkout", "confirm"],
    "checkout_with_coupon":        ["search", "add_to_cart", "begin_checkout", "apply_coupon", "confirm"],
    "checkout_with_address_entry": ["search", "add_to_cart", "begin_checkout", "enter_address", "confirm"],
    "order_tracking":              ["navigate_orders", "find_order"],
    "product_comparison":          ["search", "compare"],
}

# Baseline completion rates for each harness configuration.
# These are synthetic but consistent with the paper's reported values.
BASELINE_RATES = {
    "product_search":              0.623,
    "add_to_cart":                 0.541,
    "multi_item_cart":             0.448,
    "checkout_single_item":        0.372,
    "checkout_with_coupon":        0.318,
    "checkout_with_address_entry": 0.294,
    "order_tracking":              0.587,
    "product_comparison":          0.512,
}


@dataclass
class EpisodeResult:
    task_category: str
    harness: HarnessConfig
    success: bool
    stages_completed: list
    n_steps: int
    trace: list = field(default_factory=list)   # list of observation dicts
    failure_mode: Optional[str] = None
    elapsed_sec: float = 0.0


class MockShopGymEnv:
    """
    A synthetic ShopGym environment that simulates task completion rates
    based on the harness configuration without requiring a live browser.

    The completion rates are modeled as the product of per-stage completion
    rates, where each harness axis has a multiplicative effect on each stage.
    This approximates the deceptive landscape: some harness axes help early
    stages but hurt late stages.
    """

    # Per-axis effect on task stages (additive modifier on log-odds).
    # Larger magnitudes so harness differences translate to meaningful completion rate gaps.
    # Format: {axis_value: {stage_index: log_odds_effect}}
    AXIS_EFFECTS = {
        # obs_modality: screenshot good for visual forms; ax_tree good for structured search
        "screenshot":    {0: 0.0, 1:+0.1, 2: -0.2, 3: +0.6, 4: +0.8},  # great for visual/payment
        "ax_tree":       {0:+0.8, 1:+0.6, 2: +0.2, 3: -0.4, 4: -0.7},  # great for search, bad for checkout
        "dom_text":      {0:+0.3, 1:+0.2, 2: 0.0,  3: -0.2, 4: -0.4},
        "hybrid_ax_ss":  {0:+0.5, 1:+0.4, 2: +0.4, 3: +0.4, 4: +0.5},  # balanced & good
        "hybrid_dom_ss": {0:+0.2, 1:+0.2, 2: +0.2, 3: +0.2, 4: +0.2},

        # action_vocab
        "low_level":     {0: -0.2, 1: 0.0, 2: 0.0, 3: +0.2, 4: +0.2},
        "high_level":    {0: +0.3, 1:+0.2, 2: 0.0, 3: -0.2, 4: -0.2},
        "mixed":         {0: +0.1, 1:+0.2, 2:+0.2, 3: +0.2, 4: +0.2},

        # ctx_window
        "last_1":        {0: 0.0, 1: 0.0, 2: -0.2, 3: -0.4, 4: -0.6},
        "last_3":        {0: 0.0, 1: 0.0, 2:  0.0, 3:  0.0, 4:  0.0},  # baseline
        "last_5":        {0: 0.0, 1: 0.0, 2: +0.2, 3: +0.2, 4: +0.2},
        "full_summary":  {0:-0.1, 1: 0.0, 2: +0.3, 3: +0.5, 4: +0.6},

        # scaffold
        "none":          {0: 0.0, 1: 0.0, 2:  0.0, 3:  0.0, 4:  0.0},  # baseline
        "step_counter":  {0: 0.0, 1: 0.0, 2: +0.2, 3: +0.2, 4: +0.2},
        "error_overlay": {0: 0.0, 1: 0.0, 2: +0.2, 3: +0.4, 4: +0.4},
        "task_decomp":   {0:+0.2, 1:+0.2, 2: +0.3, 3: +0.2, 4: +0.2},
        "full":          {0:+0.2, 1:+0.2, 2: +0.4, 3: +0.6, 4: +0.7},

        # retry_policy (note: "none" key clash with scaffold — use distinct dicts per axis)
        "once":          {0: 0.0, 1: 0.0, 2:  0.0, 3:  0.0, 4:  0.0},  # baseline
        "backtrack":     {0: 0.0, 1: 0.0, 2: +0.2, 3: +0.4, 4: +0.5},
    }
    # "none" applies to both scaffold=none and retry_policy=none; same no-effect vector
    AXIS_EFFECTS["none"] = {0: 0.0, 1: 0.0, 2: -0.1, 3: -0.2, 4: -0.2}

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = random.Random(rng_seed)
        self._noise_level = 0.05  # task-level stochasticity

    def _stage_success_prob(self, task_category: str, stage_idx: int,
                             harness: HarnessConfig) -> float:
        """
        Compute per-stage success probability given harness config.
        Models the harness landscape including deceptive interactions.
        """
        baseline_q = BASELINE_RATES[task_category]
        n_stages = len(TASK_STAGES[task_category])

        # Base per-stage probability from overall baseline
        # Model: P(all stages) = prod(P(stage_k))
        # => P(stage_k) = baseline_q^(1/n_stages) roughly
        base_stage_p = baseline_q ** (1.0 / max(n_stages, 1))

        # Accumulate harness effects on this stage
        log_odds_base = math.log(base_stage_p / (1 - base_stage_p + 1e-9) + 1e-9)

        axes = [
            harness.obs_modality,
            harness.action_vocab,
            harness.ctx_window,
            harness.scaffold,
            harness.retry_policy,
        ]

        for axis_val in axes:
            effects = self.AXIS_EFFECTS.get(axis_val, {})
            effect = effects.get(stage_idx, effects.get(min(stage_idx, max(effects.keys(), default=0)), 0.0))
            log_odds_base += effect

        p = 1.0 / (1.0 + math.exp(-log_odds_base))
        # Add small noise
        p = max(0.01, min(0.99, p + self.rng.gauss(0, self._noise_level / 2)))
        return p

    def run_episode(self, task_category: str, harness: HarnessConfig) -> EpisodeResult:
        """
        Simulate a single task episode.  Each stage must succeed for the next
        to be attempted.  Returns EpisodeResult with trace info.
        """
        stages = TASK_STAGES[task_category]
        stages_completed = []
        trace = []
        failure_mode = None

        t_start = time.time()
        for i, stage in enumerate(stages):
            p = self._stage_success_prob(task_category, i, harness)
            success = self.rng.random() < p

            # Generate synthetic trace entry
            trace.append({
                "stage": stage,
                "stage_idx": i,
                "success": success,
                "obs_modality": harness.obs_modality,
                "p_success": p,
            })

            if success:
                stages_completed.append(stage)
            else:
                # Determine failure mode based on stage and harness
                if i >= 2 and harness.obs_modality in ("ax_tree", "dom_text"):
                    failure_mode = "visual_form_missing"
                elif i == 0:
                    failure_mode = "search_failed"
                elif harness.ctx_window == "last_1" and i > 1:
                    failure_mode = "context_lost"
                else:
                    failure_mode = "action_failed"
                break  # Episode ends at first stage failure

        overall_success = len(stages_completed) == len(stages)
        return EpisodeResult(
            task_category=task_category,
            harness=harness,
            success=overall_success,
            stages_completed=stages_completed,
            n_steps=len(trace) * 5 + self.rng.randint(0, 10),
            trace=trace,
            failure_mode=failure_mode if not overall_success else None,
            elapsed_sec=time.time() - t_start,
        )

    def run_n_episodes(self, task_category: str, harness: HarnessConfig, n: int) -> list:
        """Run n episodes and return list of EpisodeResult."""
        return [self.run_episode(task_category, harness) for _ in range(n)]


# ── Real Playwright-based environment (stub, requires live ShopGym server) ──

class PlaywrightShopGymEnv:
    """
    Live browser-based ShopGym environment using Playwright.
    Requires:
      1. ShopGym Flask app running at shopgym_base_url
      2. Playwright installed: `playwright install chromium`
      3. Anthropic API key set in environment
    """

    def __init__(self, shopgym_base_url: str = "http://localhost:5000",
                 model: str = "claude-sonnet-4-6",
                 agent_temperature: float = 0.0):
        self.base_url = shopgym_base_url
        self.model = model
        self.agent_temperature = agent_temperature
        self._client = None
        self._browser = None
        self._playwright = None

    def _init_anthropic(self):
        import anthropic
        import os
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _init_browser(self):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)

    def _encode_observation(self, page, harness: HarnessConfig) -> dict:
        """Encode browser state according to harness obs_modality."""
        obs = {}
        if harness.obs_modality in ("screenshot", "hybrid_ax_ss", "hybrid_dom_ss"):
            obs["screenshot"] = page.screenshot(type="png")
        if harness.obs_modality in ("ax_tree", "hybrid_ax_ss"):
            obs["ax_tree"] = page.accessibility.snapshot()
        if harness.obs_modality in ("dom_text", "hybrid_dom_ss"):
            obs["dom_text"] = page.inner_text("body")[:8000]
        if harness.scaffold in ("step_counter", "full"):
            obs["step_counter"] = True
        if harness.scaffold in ("error_overlay", "full"):
            obs["error_overlay"] = True
        if harness.scaffold in ("task_decomp", "full"):
            obs["task_decomp"] = True
        return obs

    def run_episode(self, task_category: str, harness: HarnessConfig,
                    task_params: Optional[dict] = None) -> EpisodeResult:
        """Run a single live episode. Requires live server + API key."""
        if self._client is None:
            self._init_anthropic()
        if self._browser is None:
            self._init_browser()

        context = self._browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            page.goto(f"{self.base_url}/task/{task_category}")
            result = self._run_agent_loop(page, task_category, harness, task_params)
        finally:
            context.close()

        return result

    def run_n_episodes(
        self, task_category: str, harness: HarnessConfig, n: int
    ) -> list[EpisodeResult]:
        return [self.run_episode(task_category, harness) for _ in range(n)]

    def _run_agent_loop(self, page, task_category: str,
                        harness: HarnessConfig, task_params: Optional[dict]) -> EpisodeResult:
        """Main agent-environment loop."""
        import base64
        import json

        stages = TASK_STAGES[task_category]
        trace = []
        messages = []
        max_steps = 50

        system_prompt = self._build_system_prompt(harness, task_category)

        for step in range(max_steps):
            obs = self._encode_observation(page, harness)

            # Build message content
            content = []
            if "screenshot" in obs and isinstance(obs["screenshot"], bytes):
                img_b64 = base64.b64encode(obs["screenshot"]).decode()
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
                })
            if "ax_tree" in obs:
                content.append({"type": "text", "text": f"Accessibility tree:\n{json.dumps(obs['ax_tree'], indent=2)[:3000]}"})
            if "dom_text" in obs:
                content.append({"type": "text", "text": f"Page text:\n{obs['dom_text']}"})

            messages.append({"role": "user", "content": content})

            # Manage context window
            messages = self._apply_ctx_window(messages, harness)

            # Call the frozen LLM
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                temperature=self.agent_temperature,
            )
            action_text = response.content[0].text
            messages.append({"role": "assistant", "content": action_text})

            # Execute action
            success, done, stage_complete = self._execute_action(page, action_text, harness)
            trace.append({"step": step, "action": action_text, "success": success})

            if done:
                break

        # Evaluate final task completion
        task_success = self._check_task_completion(page, task_category)
        return EpisodeResult(
            task_category=task_category,
            harness=harness,
            success=task_success,
            stages_completed=stages if task_success else [],
            n_steps=len(trace),
            trace=trace,
            failure_mode=None if task_success else "incomplete",
        )

    def _build_system_prompt(self, harness: HarnessConfig, task_category: str) -> str:
        prompt = f"You are a web agent completing e-commerce tasks. Task: {task_category}.\n"
        if harness.scaffold in ("task_decomp", "full"):
            prompt += f"Stages to complete: {', '.join(TASK_STAGES[task_category])}.\n"
        if harness.action_vocab == "high_level":
            prompt += "Use high-level actions: CLICK(label), TYPE(field, text), SELECT(option), SCROLL(direction).\n"
        elif harness.action_vocab == "low_level":
            prompt += "Use low-level actions: CLICK(x, y), TYPE(x, y, text), SCROLL(x, y, delta).\n"
        else:
            prompt += "Use high-level actions when possible, fall back to coordinates when needed.\n"
        prompt += "Output one action per response. Say DONE when the task is complete."
        return prompt

    def _apply_ctx_window(self, messages: list, harness: HarnessConfig) -> list:
        if harness.ctx_window == "last_1":
            return messages[-2:]  # keep last user+assistant pair
        elif harness.ctx_window == "last_3":
            return messages[-6:]
        elif harness.ctx_window == "last_5":
            return messages[-10:]
        return messages  # full_summary: keep all (in practice would summarize)

    def _execute_action(self, page, action_text: str, harness: HarnessConfig):
        """Parse and execute agent action. Returns (success, done, stage_complete)."""
        action_text = action_text.strip()
        if "DONE" in action_text:
            return True, True, True

        try:
            if action_text.startswith("CLICK("):
                label = action_text[6:-1].strip('"\'')
                page.get_by_text(label).first.click(timeout=3000)
                return True, False, False
            elif action_text.startswith("TYPE("):
                parts = action_text[5:-1].split(",", 1)
                field_name = parts[0].strip().strip('"\'')
                text = parts[1].strip().strip('"\'') if len(parts) > 1 else ""
                page.get_by_label(field_name).first.fill(text, timeout=3000)
                return True, False, False
        except Exception:
            if harness.retry_policy == "backtrack":
                page.go_back()
            return False, False, False

        return False, False, False

    def _check_task_completion(self, page, task_category: str) -> bool:
        """Check if task is complete based on page state."""
        url = page.url
        if task_category in ("checkout_single_item", "checkout_with_coupon",
                              "checkout_with_address_entry"):
            return "confirmation" in url or "success" in url
        elif task_category == "product_search":
            return "results" in url and page.locator(".product-card").count() > 0
        elif task_category in ("add_to_cart", "multi_item_cart"):
            cart_count = page.locator(".cart-count").text_content() or "0"
            return int(cart_count) > 0
        return False

    def close(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
