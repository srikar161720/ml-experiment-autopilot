"""ReportGenerator - Gemini-powered final experiment report generation.

This component generates a publication-ready Markdown report at the end
of an autopilot run. It uses a single Gemini call for narrative sections
(executive summary, methodology, insights, recommendations) and locally-built
structured sections (dataset overview, results table, best model, appendix).
Falls back to a template-only report if Gemini fails.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiInvalidResponseError,
)
from src.orchestration.state import ExperimentState


# System prompt for report generation
REPORT_GENERATOR_SYSTEM_PROMPT = """You are an expert ML research writer creating a professional experiment report. Your role is to synthesize experiment results into a clear, insightful narrative.

PRINCIPLES:
1. Write in a professional, technical tone suitable for data science audiences
2. Be specific — reference actual metric values, model names, and iteration numbers
3. Explain WHY certain approaches worked or failed, not just WHAT happened
4. Connect insights across experiments to tell a coherent story
5. Provide actionable recommendations based on the evidence

SECTIONS TO GENERATE:
You will be given experiment data and asked to generate narrative content for specific report sections. Generate ONLY the requested content — no extra formatting.

EXECUTIVE SUMMARY:
- One concise paragraph (3-5 sentences)
- State the problem, approach, key finding, and best result
- Mention the total number of experiments and the improvement over baseline

METHODOLOGY:
- 2-3 paragraphs describing the approach
- Mention the iterative hypothesis-driven design
- Reference the types of models explored and preprocessing strategies tried
- Note the termination reason

KEY INSIGHTS:
- 3-5 bullet points with substantive observations
- Each insight should connect a specific experimental choice to an outcome
- Include patterns discovered across iterations

RECOMMENDATIONS:
- 3-5 actionable bullet points for future work
- Based on what the experiments revealed about the data and models
- Include both quick wins and longer-term suggestions

Respond with the requested Markdown content only. No JSON wrapping."""


# Section delimiter used to parse Gemini's response into separate sections.
SECTION_DELIMITER = "---SECTION---"


class ReportGenerator:
    """Generates publication-ready Markdown reports using Gemini 3.

    Key features:
    - Single Gemini call for all narrative sections (efficient API usage)
    - Complete fallback report if Gemini fails (template-only)
    - Locally-built structured sections (tables, stats)
    - Saves report with timestamped filename

    Usage:
        generator = ReportGenerator(gemini_client)
        report_path = generator.generate(state=state, output_dir=output_dir)
    """

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the report generator.

        Args:
            gemini_client: Shared GeminiClient instance for API calls.
        """
        self.client = gemini_client

    def generate(self, state: ExperimentState, output_dir: Path) -> Path:
        """Generate the final experiment report.

        Args:
            state: Complete ExperimentState after all iterations.
            output_dir: Base output directory (reports saved to output_dir/reports/).

        Returns:
            Path to the generated Markdown report file.
        """
        # Build all local (non-Gemini) sections
        dataset_name = self._extract_dataset_name(state)
        dataset_section = self._build_dataset_section(state)
        results_table = self._build_results_table(state)
        best_model_section = self._build_best_model_section(state)
        appendix_section = self._build_appendix_section(state)
        run_metadata = self._build_run_metadata(state)

        # Attempt Gemini-generated narrative sections
        try:
            narrative = self._generate_narrative(state)
        except Exception:
            narrative = self._get_fallback_narrative(state)

        # Assemble the full report
        report = self._assemble_report(
            dataset_name=dataset_name,
            executive_summary=narrative["executive_summary"],
            dataset_section=dataset_section,
            methodology=narrative["methodology"],
            results_table=results_table,
            best_model_section=best_model_section,
            key_insights=narrative["key_insights"],
            recommendations=narrative["recommendations"],
            appendix_section=appendix_section,
            run_metadata=run_metadata,
        )

        # Save to file
        report_path = self._save_report(report, dataset_name, output_dir)
        return report_path

    # ── Gemini narrative ──────────────────────────────────────────────

    def _generate_narrative(self, state: ExperimentState) -> dict:
        """Single Gemini call to generate all narrative sections.

        Args:
            state: Complete experiment state.

        Returns:
            Dict with keys: executive_summary, methodology,
            key_insights, recommendations.

        Raises:
            GeminiError: If Gemini API call fails.
            GeminiInvalidResponseError: If response cannot be parsed.
        """
        prompt = self._build_report_prompt(state)

        response = self.client.generate(
            prompt=prompt,
            system_instruction=REPORT_GENERATOR_SYSTEM_PROMPT,
            thinking_level="medium",
            use_history=False,
        )

        return self._parse_narrative_response(response.text)

    def _build_report_prompt(self, state: ExperimentState) -> str:
        """Build the prompt for narrative generation.

        Includes condensed experiment data for Gemini to reference.

        Args:
            state: Complete experiment state.

        Returns:
            Formatted prompt string.
        """
        dataset_name = self._extract_dataset_name(state)
        primary_metric = state.config.primary_metric or "metric"
        n_successful = sum(1 for e in state.experiments if e.success)
        n_failed = len(state.experiments) - n_successful

        # Compute baseline improvement
        improvement_str = "N/A"
        baseline_metric = None
        if state.experiments and state.experiments[0].success:
            baseline_metric = state.experiments[0].metrics.get(primary_metric)
        if baseline_metric is not None and state.best_metric is not None and baseline_metric != 0:
            lower_is_better = self._is_lower_better(primary_metric)
            if lower_is_better:
                pct = ((baseline_metric - state.best_metric) / abs(baseline_metric)) * 100
            else:
                pct = ((state.best_metric - baseline_metric) / abs(baseline_metric)) * 100
            improvement_str = f"{pct:.1f}%"

        # Condensed experiment results
        experiments_data = []
        for exp in state.experiments:
            metric_val = exp.metrics.get(primary_metric)
            experiments_data.append({
                "iteration": exp.iteration,
                "name": exp.experiment_name,
                "model": exp.model_type,
                primary_metric: round(metric_val, 6) if metric_val is not None else None,
                "hypothesis": exp.hypothesis[:100] if exp.hypothesis else "",
                "success": exp.success,
            })

        # Profile summary
        profile_info = ""
        if state.data_profile:
            missing_cols = sum(
                1 for v in state.data_profile.missing_values.values() if v > 0
            )
            profile_info = (
                f"- Rows: {state.data_profile.n_rows}, Columns: {state.data_profile.n_columns}\n"
                f"- Numeric features: {len(state.data_profile.numeric_columns)}, "
                f"Categorical features: {len(state.data_profile.categorical_columns)}\n"
                f"- Target: {state.data_profile.target_column} ({state.data_profile.target_type})\n"
                f"- Columns with missing values: {missing_cols}"
            )

        best_metric_str = f"{state.best_metric:.6f}" if state.best_metric is not None else "N/A"

        prompt = f"""## Report Generation Request

### Dataset
- Name: {dataset_name}
{profile_info}
- Task: {state.config.task_type.value}
- Primary Metric: {primary_metric}

### Experiment Summary
- Total iterations: {state.current_iteration}
- Successful experiments: {n_successful}
- Failed experiments: {n_failed}
- Total time: {state.get_elapsed_time():.1f}s
- Termination reason: {state.termination_reason or "completed"}

### All Experiment Results
```json
{json.dumps(experiments_data, indent=2)}
```

### Best Result
- Experiment: {state.best_experiment or "N/A"}
- {primary_metric}: {best_metric_str}
- Improvement over baseline: {improvement_str}

### Constraints
{state.config.constraints or "None provided"}

### Your Task
Generate the following four sections for the experiment report. Separate each section with the exact delimiter `{SECTION_DELIMITER}`:

1. **Executive Summary**: One paragraph (3-5 sentences) summarizing the problem, approach, key finding, and best result.

2. **Methodology**: 2-3 paragraphs describing the iterative approach, models explored, and preprocessing strategies.

3. **Key Insights**: 3-5 bullet points (use `- ` prefix) with substantive observations connecting experimental choices to outcomes.

4. **Recommendations**: 3-5 bullet points (use `- ` prefix) with actionable next steps for future work.

Output format:
[Executive Summary text]
{SECTION_DELIMITER}
[Methodology text]
{SECTION_DELIMITER}
[Key Insights bullets]
{SECTION_DELIMITER}
[Recommendations bullets]"""

        return prompt

    def _parse_narrative_response(self, response_text: str) -> dict:
        """Parse Gemini's delimited response into four sections.

        Args:
            response_text: Raw text from Gemini.

        Returns:
            Dict with keys: executive_summary, methodology,
            key_insights, recommendations.

        Raises:
            GeminiInvalidResponseError: If response is empty or malformed.
        """
        if not response_text or not response_text.strip():
            raise GeminiInvalidResponseError("Empty response from Gemini")

        sections = response_text.split(SECTION_DELIMITER)
        sections = [s.strip() for s in sections]

        result = {
            "executive_summary": sections[0] if len(sections) > 0 else "",
            "methodology": sections[1] if len(sections) > 1 else "",
            "key_insights": sections[2] if len(sections) > 2 else "",
            "recommendations": sections[3] if len(sections) > 3 else "",
        }

        # Validate we got meaningful content
        if not result["executive_summary"] or len(result["executive_summary"]) < 50:
            raise GeminiInvalidResponseError(
                "Executive summary too short or empty"
            )

        return result

    def _get_fallback_narrative(self, state: ExperimentState) -> dict:
        """Generate template-based narrative when Gemini fails.

        Args:
            state: Complete experiment state.

        Returns:
            Dict with keys: executive_summary, methodology,
            key_insights, recommendations.
        """
        dataset_name = self._extract_dataset_name(state)
        task = state.config.task_type.value
        metric = state.config.primary_metric or "metric"
        n_experiments = len(state.experiments)
        n_successful = sum(1 for e in state.experiments if e.success)
        best = state.best_experiment or "N/A"
        best_val = f"{state.best_metric:.6f}" if state.best_metric is not None else "N/A"

        # Compute baseline improvement
        improvement_str = ""
        baseline_metric = None
        if state.experiments and state.experiments[0].success:
            baseline_metric = state.experiments[0].metrics.get(metric)
        if baseline_metric is not None and state.best_metric is not None and baseline_metric != 0:
            lower_is_better = self._is_lower_better(metric)
            if lower_is_better:
                pct = ((baseline_metric - state.best_metric) / abs(baseline_metric)) * 100
            else:
                pct = ((state.best_metric - baseline_metric) / abs(baseline_metric)) * 100
            improvement_str = (
                f"The best model achieved a {abs(pct):.1f}% "
                f"{'improvement' if pct > 0 else 'change'} over the baseline."
            )

        # Collect unique model types tried
        models_tried = list(set(
            e.model_type for e in state.experiments if e.success
        ))

        return {
            "executive_summary": (
                f"This report summarizes an automated ML experiment session on the "
                f"{dataset_name} dataset for {task}. A total of {n_experiments} experiments "
                f"were conducted ({n_successful} successful), optimizing for {metric}. "
                f"The best performing model was {best} with {metric} = {best_val}. "
                f"{improvement_str} The session terminated because: "
                f"{state.termination_reason or 'completed'}."
            ),
            "methodology": (
                f"The experiment used an iterative, hypothesis-driven approach powered by "
                f"an AI agent (Gemini 3). Starting with a baseline model, the system designed "
                f"and executed experiments sequentially, analyzing results after each iteration "
                f"to inform the next.\n\n"
                f"Models explored include: {', '.join(models_tried) if models_tried else 'N/A'}. "
                f"The session ran for {n_experiments} iterations before terminating "
                f"due to: {state.termination_reason or 'completion'}."
            ),
            "key_insights": (
                f"- {n_successful} out of {n_experiments} experiments completed successfully.\n"
                f"- The best result was achieved by {best} ({metric} = {best_val}).\n"
                f"- Models tried: {', '.join(models_tried) if models_tried else 'N/A'}.\n"
                f"- {improvement_str if improvement_str else 'Baseline comparison not available.'}"
            ),
            "recommendations": (
                f"- Consider running additional iterations with hyperparameter tuning "
                f"on the best model ({best}).\n"
                f"- Explore feature engineering to improve model performance.\n"
                f"- Try ensemble methods combining the top-performing models.\n"
                f"- Evaluate model performance on a held-out test set for production readiness."
            ),
        }

    # ── Local sections ────────────────────────────────────────────────

    def _extract_dataset_name(self, state: ExperimentState) -> str:
        """Extract human-readable dataset name from data path.

        Args:
            state: Experiment state containing config.data_path.

        Returns:
            Dataset name (stem of the file).
        """
        return Path(state.config.data_path).stem

    def _build_dataset_section(self, state: ExperimentState) -> str:
        """Build the Dataset Overview section from DataProfile.

        Args:
            state: Experiment state containing data_profile.

        Returns:
            Markdown table with dataset statistics.
        """
        if state.data_profile is None:
            return "*Data profile not available.*"

        profile = state.data_profile
        missing_cols = sum(1 for v in profile.missing_values.values() if v > 0)
        total_missing = sum(profile.missing_values.values())

        lines = [
            "| Property | Value |",
            "|----------|-------|",
            f"| Rows | {profile.n_rows:,} |",
            f"| Columns | {profile.n_columns} |",
            f"| Numeric Features | {len(profile.numeric_columns)} |",
            f"| Categorical Features | {len(profile.categorical_columns)} |",
            f"| Target Column | {profile.target_column} |",
            f"| Target Type | {profile.target_type} |",
            f"| Columns with Missing Values | {missing_cols} |",
            f"| Total Missing Values | {total_missing:,} |",
        ]

        # Add target statistics if available
        if profile.target_stats:
            lines.append("")
            lines.append("**Target Statistics:**")
            lines.append("")
            lines.append("| Statistic | Value |")
            lines.append("|-----------|-------|")
            for stat, value in profile.target_stats.items():
                if isinstance(value, float):
                    lines.append(f"| {stat} | {value:.4f} |")
                else:
                    lines.append(f"| {stat} | {value} |")

        return "\n".join(lines)

    def _build_results_table(self, state: ExperimentState) -> str:
        """Build the experiment results Markdown table.

        Args:
            state: Experiment state containing experiment results.

        Returns:
            Markdown table of all experiments.
        """
        if not state.experiments:
            return "*No experiments were conducted.*"

        primary_metric = state.config.primary_metric or "metric"
        metric_header = primary_metric.upper()

        lines = [
            f"| # | Experiment | Model | {metric_header} | Status | Hypothesis |",
            "|---|-----------|-------|------|--------|------------|",
        ]

        for i, exp in enumerate(state.experiments):
            if exp.success:
                metric_val = exp.metrics.get(primary_metric)
                metric_str = f"{metric_val:.6f}" if metric_val is not None else "--"
                status = "OK"
            else:
                metric_str = "--"
                status = "FAILED"

            # Truncate hypothesis for table readability
            hypothesis = exp.hypothesis[:60] + "..." if len(exp.hypothesis) > 60 else exp.hypothesis

            lines.append(
                f"| {i} | {exp.experiment_name} | {exp.model_type} | "
                f"{metric_str} | {status} | {hypothesis} |"
            )

        return "\n".join(lines)

    def _build_best_model_section(self, state: ExperimentState) -> str:
        """Build the Best Model details section.

        Args:
            state: Experiment state.

        Returns:
            Markdown section with best model details.
        """
        if state.best_experiment is None or state.best_metric is None:
            return "*No successful experiments to report as best model.*"

        # Find best experiment
        best_exp = None
        for exp in state.experiments:
            if exp.experiment_name == state.best_experiment:
                best_exp = exp
                break

        if best_exp is None:
            return "*Best experiment data not found.*"

        primary_metric = state.config.primary_metric or "metric"

        lines = [
            f"**Model**: {best_exp.model_type}",
            f"**Experiment**: {best_exp.experiment_name}",
            f"**Iteration**: {best_exp.iteration}",
            f"**Primary Metric ({primary_metric})**: {state.best_metric:.6f}",
            "",
        ]

        # All metrics table
        if best_exp.metrics:
            lines.append("**All Metrics:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric_name, metric_val in sorted(best_exp.metrics.items()):
                lines.append(f"| {metric_name} | {metric_val:.6f} |")
            lines.append("")

        # Hyperparameters table
        if best_exp.model_params:
            lines.append("**Hyperparameters:**")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for param, value in sorted(best_exp.model_params.items()):
                lines.append(f"| {param} | {value} |")
            lines.append("")

        # Hypothesis
        if best_exp.hypothesis:
            lines.append(f"**Hypothesis**: {best_exp.hypothesis}")

        return "\n".join(lines)

    def _build_appendix_section(self, state: ExperimentState) -> str:
        """Build the Appendix with per-experiment details.

        Args:
            state: Experiment state.

        Returns:
            Markdown section with detailed experiment logs.
        """
        if not state.experiments:
            return "*No experiments to display.*"

        blocks = []
        for i, exp in enumerate(state.experiments):
            lines = [
                f"#### Experiment {i}: {exp.experiment_name}",
                f"- **Model**: {exp.model_type}",
                f"- **Hypothesis**: {exp.hypothesis}",
                f"- **Status**: {'Success' if exp.success else 'Failed'}",
            ]

            if exp.success and exp.metrics:
                metrics_str = ", ".join(
                    f"{k}={v:.6f}" for k, v in sorted(exp.metrics.items())
                )
                lines.append(f"- **Metrics**: {metrics_str}")

            if not exp.success and exp.error_message:
                lines.append(f"- **Error**: {exp.error_message[:200]}")

            if exp.execution_time > 0:
                lines.append(f"- **Execution Time**: {exp.execution_time:.1f}s")

            if exp.reasoning:
                reasoning_short = exp.reasoning[:200]
                if len(exp.reasoning) > 200:
                    reasoning_short += "..."
                lines.append(f"- **Reasoning**: {reasoning_short}")

            blocks.append("\n".join(lines))

        return "\n\n---\n\n".join(blocks)

    def _build_run_metadata(self, state: ExperimentState) -> str:
        """Build run metadata footer.

        Args:
            state: Experiment state.

        Returns:
            Markdown footer with session info.
        """
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = state.get_elapsed_time()

        return (
            f"---\n\n"
            f"*Generated by ML Experiment Autopilot*  \n"
            f"*Session ID: {state.session_id} | "
            f"Date: {date_str} | "
            f"Runtime: {elapsed:.1f}s*  \n"
            f"*Powered by Gemini 3*"
        )

    # ── Assembly ──────────────────────────────────────────────────────

    def _assemble_report(
        self,
        dataset_name: str,
        executive_summary: str,
        dataset_section: str,
        methodology: str,
        results_table: str,
        best_model_section: str,
        key_insights: str,
        recommendations: str,
        appendix_section: str,
        run_metadata: str,
    ) -> str:
        """Assemble all sections into the final Markdown report.

        Returns:
            Complete Markdown report as a string.
        """
        return f"""# ML Experiment Report: {dataset_name}

## Executive Summary

{executive_summary}

## Dataset Overview

{dataset_section}

## Methodology

{methodology}

## Experiment Results

### Performance Summary

{results_table}

### Best Model

{best_model_section}

## Key Insights

{key_insights}

## Recommendations

{recommendations}

## Appendix

### Experiment Details

{appendix_section}

{run_metadata}
"""

    def _save_report(
        self,
        report: str,
        dataset_name: str,
        output_dir: Path,
    ) -> Path:
        """Save the report to a file and return the path.

        Args:
            report: Complete Markdown report content.
            dataset_name: Name of the dataset (for filename).
            output_dir: Base output directory.

        Returns:
            Path to the saved report file.
        """
        reports_dir = Path(output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{dataset_name}_{timestamp}.md"
        report_path = reports_dir / filename
        report_path.write_text(report)

        return report_path

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_lower_better(metric_name: str) -> bool:
        """Check if a metric is one where lower values are better.

        Args:
            metric_name: Name of the metric.

        Returns:
            True if lower is better (e.g., RMSE, MAE).
        """
        lower_better = ["rmse", "mse", "mae", "log_loss", "error"]
        return any(m in metric_name.lower() for m in lower_better)
