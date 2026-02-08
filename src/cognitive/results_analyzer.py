"""ResultsAnalyzer - Gemini-powered experiment results analysis.

This component analyzes experiment outcomes, compares metrics against
baselines and historical results, and detects performance patterns.
"""

import json
from typing import Optional

from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiInvalidResponseError,
)
from src.orchestration.state import (
    DataProfile,
    ExperimentResult,
    ExperimentState,
    AnalysisResult,
    MetricComparison,
    TrendPattern,
)


# System prompt for results analysis
RESULTS_ANALYZER_SYSTEM_PROMPT = """You are an expert ML results analyst. Your role is to analyze experiment outcomes and provide actionable insights.

PRINCIPLES:
1. Compare current results against baseline, best, and previous experiments
2. Identify meaningful improvements versus noise
3. Detect patterns in model performance across iterations
4. Consider both the primary metric and secondary metrics
5. Note any anomalies or unexpected results
6. Provide observations that inform next experiment design

METRIC INTERPRETATION:
- Lower is better: RMSE, MSE, MAE, log_loss, error
- Higher is better: accuracy, f1, r2, precision, recall, AUC

PATTERN DETECTION:
- "improving": Consistent improvement over last 2+ iterations
- "degrading": Consistent degradation over last 2+ iterations
- "plateau": Less than 0.5% change for 2+ iterations
- "fluctuating": Alternating improvement/degradation
- "initial": Fewer than 3 experiments to establish pattern

KEY OBSERVATIONS SHOULD:
- Be specific and actionable
- Reference actual metric values
- Connect preprocessing/model choices to outcomes
- Highlight surprising or noteworthy findings

Always respond with valid JSON matching the required schema."""


class ResultsAnalyzer:
    """Analyzes ML experiment results using Gemini 3.

    Key features:
    - Compares metrics against baseline, best, and previous
    - Detects performance trends (improving, degrading, plateau)
    - Handles lower-is-better vs higher-is-better metrics correctly
    - Generates key observations for next iteration
    - Maintains conversation context for Thought Signature continuity

    Usage:
        analyzer = ResultsAnalyzer(gemini_client)
        analysis = analyzer.analyze(
            current_result=result,
            state=experiment_state,
        )
    """

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the results analyzer.

        Args:
            gemini_client: Shared GeminiClient instance for API calls.
        """
        self.client = gemini_client
        self._analysis_count = 0

    def analyze(
        self,
        current_result: ExperimentResult,
        state: ExperimentState,
    ) -> AnalysisResult:
        """Analyze the results of a completed experiment.

        Args:
            current_result: The experiment result to analyze.
            state: Current experiment state with history.

        Returns:
            AnalysisResult with comparisons and observations.
        """
        self._analysis_count += 1

        # Compute metric comparison locally (doesn't need Gemini)
        primary_metric_name = state.config.primary_metric or "rmse"
        metric_comparison = self._compute_metric_comparison(
            current_result, state, primary_metric_name
        )

        # Detect trend pattern locally
        trend_pattern = self._detect_trend_pattern(state, primary_metric_name)

        # If experiment failed, return basic analysis without Gemini
        if not current_result.success:
            return AnalysisResult(
                experiment_name=current_result.experiment_name,
                iteration=current_result.iteration,
                success=False,
                primary_metric=metric_comparison,
                trend_pattern=trend_pattern,
                key_observations=[
                    f"Experiment failed: {current_result.error_message or 'Unknown error'}"
                ],
                reasoning="Experiment did not complete successfully.",
            )

        # Use Gemini to generate observations
        prompt = self._build_analysis_prompt(current_result, state, metric_comparison)

        try:
            response = self.client.generate_json(
                prompt=prompt,
                system_instruction=RESULTS_ANALYZER_SYSTEM_PROMPT,
                thinking_level="high",
            )

            return self._parse_analysis_response(
                response, current_result, metric_comparison, trend_pattern
            )

        except (GeminiInvalidResponseError, GeminiError):
            # Fallback to basic analysis
            return self._get_fallback_analysis(
                current_result, metric_comparison, trend_pattern
            )

    def _compute_metric_comparison(
        self,
        current_result: ExperimentResult,
        state: ExperimentState,
        primary_metric: str,
    ) -> Optional[MetricComparison]:
        """Compute metric comparisons locally.

        Args:
            current_result: Current experiment result.
            state: Experiment state with history.
            primary_metric: Name of primary metric.

        Returns:
            MetricComparison with all computed values.
        """
        current_value = current_result.metrics.get(primary_metric)
        if current_value is None:
            return None

        # Get baseline (first experiment, usually iteration 0)
        baseline_value = None
        if state.experiments:
            baseline = state.experiments[0]
            if baseline.success:
                baseline_value = baseline.metrics.get(primary_metric)

        # Get best value
        best_value = state.best_metric

        # Get previous value
        previous_value = None
        if len(state.experiments) >= 2:
            prev_exp = state.experiments[-2]
            if prev_exp.success:
                previous_value = prev_exp.metrics.get(primary_metric)

        # Calculate percentage changes
        lower_is_better = self._is_lower_better(primary_metric)

        change_from_baseline = None
        if baseline_value is not None and abs(baseline_value) > 1e-12:
            if lower_is_better:
                change_from_baseline = ((baseline_value - current_value) / abs(baseline_value)) * 100
            else:
                change_from_baseline = ((current_value - baseline_value) / abs(baseline_value)) * 100

        change_from_best = None
        if best_value is not None and abs(best_value) > 1e-12:
            if lower_is_better:
                change_from_best = ((best_value - current_value) / abs(best_value)) * 100
            else:
                change_from_best = ((current_value - best_value) / abs(best_value)) * 100

        change_from_previous = None
        if previous_value is not None and abs(previous_value) > 1e-12:
            if lower_is_better:
                change_from_previous = ((previous_value - current_value) / abs(previous_value)) * 100
            else:
                change_from_previous = ((current_value - previous_value) / abs(previous_value)) * 100

        # Determine if improvement
        improvement_threshold = state.config.improvement_threshold * 100  # Convert to percentage
        is_improvement = False
        is_new_best = False

        if change_from_best is not None:
            is_improvement = change_from_best > improvement_threshold
            is_new_best = is_improvement
        elif best_value is None:
            is_new_best = True
            is_improvement = True

        return MetricComparison(
            metric_name=primary_metric,
            current_value=current_value,
            baseline_value=baseline_value,
            best_value=best_value,
            previous_value=previous_value,
            change_from_baseline_pct=change_from_baseline,
            change_from_best_pct=change_from_best,
            change_from_previous_pct=change_from_previous,
            is_improvement=is_improvement,
            is_new_best=is_new_best,
        )

    def _detect_trend_pattern(
        self,
        state: ExperimentState,
        primary_metric: str,
    ) -> TrendPattern:
        """Detect the current performance trend pattern.

        Args:
            state: Experiment state with history.
            primary_metric: Name of primary metric.

        Returns:
            TrendPattern enum value.
        """
        # Need at least 3 experiments to detect a trend
        if len(state.experiments) < 3:
            return TrendPattern.INITIAL

        # Get last 3 successful experiments' metrics
        recent_metrics = []
        for exp in reversed(state.experiments):
            if exp.success:
                value = exp.metrics.get(primary_metric)
                if value is not None:
                    recent_metrics.append(value)
                if len(recent_metrics) >= 3:
                    break

        if len(recent_metrics) < 3:
            return TrendPattern.INITIAL

        # Reverse to chronological order
        recent_metrics = list(reversed(recent_metrics))
        lower_is_better = self._is_lower_better(primary_metric)
        threshold = state.config.improvement_threshold

        # Calculate changes
        changes = []
        for i in range(1, len(recent_metrics)):
            prev = recent_metrics[i - 1]
            curr = recent_metrics[i]
            if abs(prev) > 1e-12:
                if lower_is_better:
                    change = (prev - curr) / abs(prev)
                else:
                    change = (curr - prev) / abs(prev)
                changes.append(change)

        if not changes:
            return TrendPattern.INITIAL

        # Determine pattern
        all_improving = all(c > threshold for c in changes)
        all_degrading = all(c < -threshold for c in changes)
        all_plateau = all(abs(c) <= threshold for c in changes)

        if all_improving:
            return TrendPattern.IMPROVING
        elif all_degrading:
            return TrendPattern.DEGRADING
        elif all_plateau:
            return TrendPattern.PLATEAU
        else:
            return TrendPattern.FLUCTUATING

    def _is_lower_better(self, metric_name: str) -> bool:
        """Determine if lower values are better for this metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            True if lower is better.
        """
        lower_is_better = ["rmse", "mse", "mae", "log_loss", "error"]
        return any(m in metric_name.lower() for m in lower_is_better)

    def _build_analysis_prompt(
        self,
        current_result: ExperimentResult,
        state: ExperimentState,
        metric_comparison: Optional[MetricComparison],
    ) -> str:
        """Build the prompt for results analysis.

        Args:
            current_result: Current experiment result.
            state: Experiment state.
            metric_comparison: Computed metric comparison.

        Returns:
            Formatted prompt string.
        """
        # Summarize current result
        result_summary = {
            "experiment_name": current_result.experiment_name,
            "iteration": current_result.iteration,
            "model_type": current_result.model_type,
            "metrics": current_result.metrics,
            "hypothesis": current_result.hypothesis,
            "success": current_result.success,
        }

        # Summarize metric comparison
        comparison_summary = None
        if metric_comparison:
            comparison_summary = {
                "metric": metric_comparison.metric_name,
                "current": metric_comparison.current_value,
                "baseline": metric_comparison.baseline_value,
                "best": metric_comparison.best_value,
                "previous": metric_comparison.previous_value,
                "change_from_baseline_pct": metric_comparison.change_from_baseline_pct,
                "change_from_best_pct": metric_comparison.change_from_best_pct,
                "is_improvement": metric_comparison.is_improvement,
                "is_new_best": metric_comparison.is_new_best,
            }

        # Get experiment history summary (last 5)
        history = []
        for exp in state.experiments[-5:]:
            history.append({
                "name": exp.experiment_name,
                "model": exp.model_type,
                "metrics": exp.metrics,
                "success": exp.success,
            })

        prompt = f"""## Results Analysis Request

### Current Experiment
```json
{json.dumps(result_summary, indent=2)}
```

### Metric Comparison
```json
{json.dumps(comparison_summary, indent=2) if comparison_summary else "N/A"}
```

### Experiment History (last {len(history)})
```json
{json.dumps(history, indent=2)}
```

### Task Type
{state.config.task_type.value}

### Your Task
Analyze the current experiment results and provide key observations. Focus on:
1. Why did this experiment perform as it did?
2. What patterns do you see across the experiment history?
3. What actionable insights can inform the next experiment?

Respond with valid JSON:
```json
{{
    "key_observations": [
        "Observation 1 with specific metric values",
        "Observation 2 about patterns",
        "Observation 3 about actionable insights"
    ],
    "reasoning": "Detailed explanation of the analysis"
}}
```"""

        return prompt

    def _parse_analysis_response(
        self,
        response: dict,
        current_result: ExperimentResult,
        metric_comparison: Optional[MetricComparison],
        trend_pattern: TrendPattern,
    ) -> AnalysisResult:
        """Parse Gemini's analysis response.

        Args:
            response: Parsed JSON response from Gemini.
            current_result: Current experiment result.
            metric_comparison: Computed metric comparison.
            trend_pattern: Detected trend pattern.

        Returns:
            AnalysisResult with all data.
        """
        key_observations = response.get("key_observations", [])
        reasoning = response.get("reasoning", "")

        return AnalysisResult(
            experiment_name=current_result.experiment_name,
            iteration=current_result.iteration,
            success=current_result.success,
            primary_metric=metric_comparison,
            trend_pattern=trend_pattern,
            key_observations=key_observations,
            reasoning=reasoning,
        )

    def _get_fallback_analysis(
        self,
        current_result: ExperimentResult,
        metric_comparison: Optional[MetricComparison],
        trend_pattern: TrendPattern,
    ) -> AnalysisResult:
        """Get fallback analysis when Gemini fails.

        Args:
            current_result: Current experiment result.
            metric_comparison: Computed metric comparison.
            trend_pattern: Detected trend pattern.

        Returns:
            Basic AnalysisResult.
        """
        observations = []

        if metric_comparison:
            if metric_comparison.is_new_best:
                observations.append(
                    f"New best {metric_comparison.metric_name}: {metric_comparison.current_value:.6f}"
                )
            elif metric_comparison.is_improvement:
                observations.append(
                    f"Improvement in {metric_comparison.metric_name}: {metric_comparison.current_value:.6f}"
                )
            else:
                observations.append(
                    f"No improvement in {metric_comparison.metric_name}: {metric_comparison.current_value:.6f}"
                )

            if metric_comparison.change_from_baseline_pct is not None:
                direction = "better" if metric_comparison.change_from_baseline_pct > 0 else "worse"
                observations.append(
                    f"{abs(metric_comparison.change_from_baseline_pct):.1f}% {direction} than baseline"
                )

        observations.append(f"Trend pattern: {trend_pattern.value}")
        observations.append(f"Model used: {current_result.model_type}")

        return AnalysisResult(
            experiment_name=current_result.experiment_name,
            iteration=current_result.iteration,
            success=current_result.success,
            primary_metric=metric_comparison,
            trend_pattern=trend_pattern,
            key_observations=observations,
            reasoning="Fallback analysis (Gemini unavailable)",
        )

    def get_analysis_count(self) -> int:
        """Get the number of analyses performed."""
        return self._analysis_count
