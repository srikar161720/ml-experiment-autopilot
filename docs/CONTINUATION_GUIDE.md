# ML Experiment Autopilot - New Chat Continuation Guide

## 🎯 Purpose of This Document

This document helps quickly understand context and continue development of the ML Experiment Autopilot project without losing momentum.

**Read this file first**, then refer to the other documents as needed.

---

## 📍 Where We Left Off

### Project Status: Planning Complete, Ready for Implementation

**Completed:**
- ✅ Hackathon requirements analysis
- ✅ Project ideation and selection
- ✅ High-level architecture design
- ✅ Component specifications
- ✅ Implementation timeline (23 days)
- ✅ Tech stack selection
- ✅ Demo strategy
- ✅ All architectural decisions finalized
- ✅ CLI interface designed
- ✅ Testing strategy defined
- ✅ CLAUDE.md created for Claude Code

**Not Started:**
- ❌ Writing actual code
- ❌ Setting up the repository
- ❌ Testing any components

### Immediate Next Step
**Begin Phase 1: Foundation** — Set up the project structure and get a basic end-to-end flow working.

---

## 📁 Document Overview

| Document | Purpose | When to Reference |
|----------|---------|-------------------|
| `CLAUDE.md` | Claude Code context, patterns, quick reference | **Primary reference during coding** |
| `PROJECT_BRIEF.md` | High-level context, hackathon details | Understanding "why" and constraints |
| `TECHNICAL_SPECIFICATION.md` | Detailed architecture, component specs | Designing/implementing components |
| `IMPLEMENTATION_PLAN.md` | Phased timeline, tasks, milestones | Planning work, tracking progress |
| `TECHNICAL_REFERENCE.md` | Code snippets, dependencies, templates | Actually writing code |
| `CONTINUATION_GUIDE.md` (this file) | Quick orientation | Starting a new session |

---

## ✅ Finalized Decisions

All major architectural decisions have been made:

| Decision | Final Choice |
|----------|--------------|
| **Execution Environment** | Subprocess (Docker as stretch goal) |
| **State Persistence** | JSON files with Pydantic validation |
| **Report Format** | Markdown only |
| **MLflow Hosting** | Local |
| **CLI Framework** | Typer |
| **Demo Datasets** | House Prices → Titanic → Credit Card (stretch) |
| **Problem Type** | User-specified via `--task` argument |
| **Metric Selection** | Gemini decides if not in constraints |
| **Preprocessing** | Gemini decides per experiment |
| **Class Imbalance** | Out of scope (stretch only) |

---

## 👤 Developer Context

**Who is building this:**
- Master's student in Data Science and Analytics (near completion)
- No prior internship/industry experience
- Goal: Build impressive portfolio for AI/ML career
- Primary language: Python
- Focus: Data Science, Machine Learning, AI

**Available resources:**
- Gemini Pro subscription (full API access)
- Google Antigravity IDE (with Gemini integration)
- Google Colab Pro (GPU access)

**Development Workflow:**
- **Claude Code** (MacOS Claude app) for code creation/editing
- **Antigravity IDE** for manual interaction and testing
- **GitHub** for version control

---

## 🏆 Hackathon Context

**Event**: Gemini 3 Hackathon by Google DeepMind & Devpost  
**Deadline**: February 9, 2026 @ 5:00pm PST  
**Prize Pool**: $100,000  
**Target Track**: "The Marathon Agent"

**Submission Requirements:**
- 3-min video demonstration
- Public GitHub repository
- Working demo
- ~200 word description of Gemini integration

---

## 🔧 Project Summary

### One-Liner
An autonomous agent that designs, executes, and iterates on ML experiments without human supervision, using Gemini 3's Thought Signatures to maintain reasoning continuity.

### CLI Interface

```bash
# Minimal
autopilot run --data train.csv --target SalePrice --task regression

# Full
autopilot run \
  --data train.csv \
  --target SalePrice \
  --task regression \
  --constraints constraints.md \
  --max-iterations 15 \
  --time-budget 3600 \
  --output-dir ./my_experiment \
  --verbose
```

### Core Components
1. **Orchestration Layer** — State machine, experiment lifecycle
2. **Cognitive Core** — Gemini 3 for design, analysis, hypothesis
3. **Execution Layer** — Data profiling, code generation, subprocess running
4. **Persistence Layer** — MLflow tracking, JSON state, artifacts

---

## 🗓️ Implementation Timeline

**Total time**: ~23 days (now through Feb 9)

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| 1. Foundation | 1-7 | Basic end-to-end flow | **START HERE** |
| 2. Intelligence | 8-14 | Gemini reasoning integration | Not started |
| 3. Robustness | 15-21 | Error handling, multi-model, reports | Not started |
| 4. Demo | 22-25 | Video, polish, submission | Not started |

---

## 💡 Key Technical Notes

### Gemini 3 API Critical Points
- **Temperature MUST be 1.0** — lower values degrade reasoning
- **Thought Signatures required** for function calling
- SDK handles signatures automatically in chat interface
- Use `thinking_level="high"` for complex reasoning tasks

### Termination Criteria Defaults
- Max iterations: 20
- Plateau threshold: 3 consecutive no-improvement
- Improvement threshold: 0.5% relative
- Time budget: 3600 seconds (1 hour)

### Testing Priority
1. Data profiler on House Prices dataset
2. Code generation produces valid Python
3. Experiment execution returns metrics
4. Full loop completes 3 iterations
5. MLflow logging works

---

## 🚀 How to Continue

### Option A: Start Implementation
> "Let's start implementing the ML Experiment Autopilot. Begin with Phase 1, Day 1: setting up the project structure and Gemini integration."

### Option B: Review Architecture
> "Walk me through the architecture before we start coding."

### Option C: Specific Component
> "Let's implement [specific component] first."

---

## 📚 Quick Reference

### Project Structure
```
ml-experiment-autopilot/
├── CLAUDE.md              # Claude Code context
├── README.md
├── requirements.txt
├── docs/                  # Planning documents
├── src/
│   ├── main.py           # CLI entry point
│   ├── config.py
│   ├── orchestration/
│   ├── cognitive/
│   ├── execution/
│   └── persistence/
├── templates/            # Code generation templates
├── tests/
├── scripts/
├── data/
└── outputs/
```

### Key Dependencies
```
google-genai>=1.0.0
scikit-learn>=1.3.0
mlflow>=2.9.0
pandas>=2.1.0
pydantic>=2.5.0
typer>=0.9.0
rich>=13.7.0
jinja2>=3.1.0
```

### First Commands
```bash
mkdir ml-experiment-autopilot
cd ml-experiment-autopilot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚠️ Important Reminders

1. **Deadline is firm**: February 9, 2026 — no extensions
2. **Scope discipline**: Core features first; "nice to haves" can be cut
3. **Demo reliability**: A working simple demo beats a broken complex one
4. **Thought Signatures**: Key differentiator — make them visible in demo
5. **CLAUDE.md**: Keep this file updated as the primary Claude Code reference

---

*Continuation Guide v2.0*
*Updated: January 2026*
*All decisions finalized*
