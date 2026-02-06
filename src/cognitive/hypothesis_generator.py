"""HypothesisGenerator - Gemini-powered hypothesis generation for next experiments.

This component analyzes experiment results and generates ranked, testable
hypotheses to guide the next iteration of the experiment loop.
"""

import json
from typing import Optional

from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiInvalidResponseError,
)
from src.orchestration.state import (
    AnalysisResult,
    ExperimentState,
    Hypothesis,
    HypothesisSet,
)


# System prompt for hypothesis generation
HYPOTHESIS_GENERATOR_SYSTEM_PROMPT = """You are an expert ML researcher generating testable hypotheses. Your role is to analyze experiment results and propose the most promising directions for the next iteration.

PRINCIPLES:
1. Generate 2-3 ranked hypotheses based on the analysis
2. Each hypothesis should be specific and testable in a single experiment
3. Consider both exploration (trying new approaches) and exploitation (refining what works)
4. Reference specific metric values and patterns from the analysis
5. Suggest concrete model choices and hyperparameters when possible
6. Balance risk — include at least one "safe" refinement and one "exploratory" option

EXPLORATION VS EXPLOITATION:
- "explore": Early iterations or after plateau — try fundamentally different approaches
- "exploit": When a promising direction is found — refine parameters and preprocessing
- "balanced": Default — mix of both strategies

CONFIDENCE SCORING:
- 0.8-1.0: Strong evidence from multiple iterations supports this direction
- 0.5-0.7: Moderate evidence, reasonable next step
- 0.2-0.4: Speculative but potentially high-value

PRIORITY:
- 1: Highest priority — implement this first
- 2: Secondary option
- 3: Exploratory/backup option

Always respond with valid JSON matching the required schema."""


class HypothesisGenerator:
    """Generates ranked hypotheses for the next experiment iteration using Gemini 3.

    Key features:
    - Generates 2-3 testable hypotheses from analysis results
    - Balances exploration vs exploitation strategies
    - Provides confidence scores and priorities
    - Suggests concrete model choices and parameters
    - Maintains conversation context for Thought Signature continuity

    Usage:
        generator = HypothesisGenerator(gemini_client)
        hypotheses = generator.generate(
            analysis=analysis_result,
            state=experiment_state,
        )
    """

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the hypothesis generator.

        Args:
            gemini_client: Shared GeminiClient instance for API calls.
        """
        self.client = gemini_client
        self._generation_count = 0

    def generate(
        self,
        analysis: AnalysisResult,
        state: ExperimentState,
    ) -> HypothesisSet:
        """Generate hypotheses for the next experiment iteration.

        Args:
            analysis: Analysis of the most recent experiment.
            state: Current experiment state with history.

        Returns:
            HypothesisSet with ranked hypotheses.
        """
        self._generation_count += 1

        # If the experiment failed, return a simple retry hypothesis
        if not analysis.success:
            return HypothesisSet(
                iteration=analysis.iteration,
                analysis_summary=f"Experiment '{analysis.experiment_name}' failed.",
                hypotheses=[
                    Hypothesis(
                        hypothesis_id="h1",
                        statement="Retry with a simpler, more robust model configuration",
                        rationale="Previous experiment failed; a simpler approach is less likely to encounter errors",
                        suggested_model="RandomForestRegressor"
                        if state.config.task_type.value == "regression"
                        else "RandomForestClassifier",
                        confidence_score=0.6,
                        priority=1,
                    )
                ],
                exploration_vs_exploitation="exploit",
                reasoning="Previous experiment failed. Falling back to a robust default.",
            )

        # Build prompt and call Gemini
        prompt = self._build_hypothesis_prompt(analysis, state)

        try:
            response = self.client.generate_json(
                prompt=prompt,
                system_instruction=HYPOTHESIS_GENERATOR_SYSTEM_PROMPT,
                thinking_level="high",
            )

            return self._parse_hypothesis_response(response, analysis)

        except (GeminiInvalidResponseError, GeminiError):
            return self._get_fallback_hypotheses(analysis, state)

    def _build_hypothesis_prompt(
        self,
        analysis: AnalysisResult,
        state: ExperimentState,
    ) -> str:
        """Build the prompt for hypothesis generation.

        Args:
            analysis: Analysis of the current experiment.
            state: Experiment state with history.

        Returns:
            Formatted prompt string.
        """
        # Serialize analysis
        analysis_summary = {
            "experiment_name": analysis.experiment_name,
            "iteration": analysis.iteration,
            "success": analysis.success,
            "trend_pattern": analysis.trend_pattern.value,
            "key_observations": analysis.key_observations,
            "reasoning": analysis.reasoning,
        }

        if analysis.primary_metric:
            analysis_summary["primary_metric"] = {
                "name": analysis.primary_metric.metric_name,
                "current": analysis.primary_metric.current_value,
                "baseline": analysis.primary_metric.baseline_value,
                "best": analysis.primary_metric.best_value,
                "change_from_baseline_pct": analysis.primary_metric.change_from_baseline_pct,
                "is_new_best": analysis.primary_metric.is_new_best,
            }

        # Get experiment history (last 5)
        history = []
        for exp in state.experiments[-5:]:
            history.append({
                "name": exp.experiment_name,
                "iteration": exp.iteration,
                "model": exp.model_type,
                "metrics": exp.metrics,
                "success": exp.success,
                "hypothesis": exp.hypothesis,
            })

        # Determine strategy hint
        iteration = analysis.iteration
        trend = analysis.trend_pattern.value
        if trend == "plateau" or iteration > state.config.max_iterations * 0.7:
            strategy_hint = "Consider more exploratory hypotheses — we may be in a local optimum."
        elif trend == "improving" and iteration <= 5:
            strategy_hint = "The current direction is promising — consider both refining it and trying alternatives."
        else:
            strategy_hint = "Balance exploration and exploitation."

        prompt = f"""## Hypothesis Generation Request

### Analysis of Latest Experiment
```json
{json.dumps(analysis_summary, indent=2)}
```

### Experiment History (last {len(history)})
```json
{json.dumps(history, indent=2)}
```

### Context
- Task type: {state.config.task_type.value}
- Primary metric: {state.config.primary_metric or "not set"}
- Current iteration: {iteration} / {state.config.max_iterations}
- Iterations without improvement: {state.iterations_without_improvement}
- Constraints: {state.config.constraints or "None"}

### Strategy Guidance
{strategy_hint}

### Your Task
Generate 2-3 ranked hypotheses for the next experiment. Each hypothesis should be:
1. Specific and testable in a single experiment
2. Connected to observations from the analysis
3. Include a concrete model suggestion and key parameters

Respond with valid JSON:
```json
{{
    "analysis_summary": "Brief summary of what the analysis tells us",
    "exploration_vs_exploitation": "explore|exploit|balanced",
    "hypotheses": [
        {{
            "hypothesis_id": "h1",
            "statement": "Clear, testable hypothesis",
            "rationale": "Why this is worth testing based on analysis",
            "suggested_model": "ModelClassName",
            "suggested_params": {{}},
            "confidence_score": 0.7,
            "priority": 1
        }}
    ],
    "reasoning": "Overall reasoning for these hypotheses"
}}
```"""

        return prompt

    def _parse_hypothesis_response(
        self,
        response: dict,
        analysis: AnalysisResult,
    ) -> HypothesisSet:
        """Parse Gemini's hypothesis response into a HypothesisSet.

        Args:
            response: Parsed JSON response from Gemini.
            analysis: The analysis that prompted this generation.

        Returns:
            HypothesisSet with parsed hypotheses.
        """
        analysis_summary = response.get("analysis_summary", "")
        exploration_vs_exploitation = response.get("exploration_vs_exploitation", "balanced")
        reasoning = response.get("reasoning", "")

        hypotheses = []
        for h_data in response.get("hypotheses", []):
            # Clamp values to valid ranges
            confidence = max(0.0, min(1.0, float(h_data.get("confidence_score", 0.5))))
            priority = max(1, min(3, int(h_data.get("priority", 2))))

            hypothesis = Hypothesis(
                hypothesis_id=h_data.get("hypothesis_id", f"h{len(hypotheses) + 1}"),
                statement=h_data.get("statement", ""),
                rationale=h_data.get("rationale", ""),
                suggested_model=h_data.get("suggested_model"),
                suggested_params=h_data.get("suggested_params", {}),
                confidence_score=confidence,
                priority=priority,
            )
            hypotheses.append(hypothesis)

        return HypothesisSet(
            iteration=analysis.iteration,
            analysis_summary=analysis_summary,
            hypotheses=hypotheses,
            exploration_vs_exploitation=exploration_vs_exploitation,
            reasoning=reasoning,
        )

    def _get_fallback_hypotheses(
        self,
        analysis: AnalysisResult,
        state: ExperimentState,
    ) -> HypothesisSet:
        """Generate fallback hypotheses when Gemini is unavailable.

        Rotates through different model suggestions based on iteration.

        Args:
            analysis: The analysis result.
            state: Current experiment state.

        Returns:
            HypothesisSet with deterministic fallback hypotheses.
        """
        is_regression = state.config.task_type.value == "regression"

        # Model rotation based on iteration
        if is_regression:
            model_options = [
                ("RandomForestRegressor", {"n_estimators": 200, "max_depth": 10}),
                ("GradientBoostingRegressor", {"n_estimators": 150, "learning_rate": 0.1}),
                ("Ridge", {"alpha": 1.0}),
            ]
        else:
            model_options = [
                ("RandomForestClassifier", {"n_estimators": 200, "max_depth": 10}),
                ("GradientBoostingClassifier", {"n_estimators": 150, "learning_rate": 0.1}),
                ("LogisticRegression", {"C": 1.0, "max_iter": 1000}),
            ]

        # Pick primary and secondary based on iteration
        idx = analysis.iteration % len(model_options)
        alt_idx = (analysis.iteration + 1) % len(model_options)

        primary_model, primary_params = model_options[idx]
        alt_model, alt_params = model_options[alt_idx]

        hypotheses = [
            Hypothesis(
                hypothesis_id="h1",
                statement=f"Try {primary_model} with tuned hyperparameters",
                rationale="Systematic exploration of model space with robust defaults",
                suggested_model=primary_model,
                suggested_params=primary_params,
                confidence_score=0.5,
                priority=1,
            ),
            Hypothesis(
                hypothesis_id="h2",
                statement=f"Try {alt_model} as an alternative approach",
                rationale="Diversifying model selection to find better fit",
                suggested_model=alt_model,
                suggested_params=alt_params,
                confidence_score=0.4,
                priority=2,
            ),
        ]

        return HypothesisSet(
            iteration=analysis.iteration,
            analysis_summary="Fallback hypothesis generation (Gemini unavailable)",
            hypotheses=hypotheses,
            exploration_vs_exploitation="balanced",
            reasoning="Fallback hypotheses (Gemini unavailable)",
        )

    def get_generation_count(self) -> int:
        """Get the number of hypothesis generations performed."""
        return self._generation_count
