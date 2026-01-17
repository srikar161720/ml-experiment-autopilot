# ML Experiment Autopilot - Project Brief

## Document Purpose
This document provides high-level context for the ML Experiment Autopilot project, including hackathon details, developer background, and project goals.

---

## 1. Developer Profile

### Background
- **Current Status**: Master's student in Data Science and Analytics, nearing program completion
- **Experience Level**: No prior internship or industry experience
- **Goal**: Build an impressive portfolio to launch AI/ML and data science career
- **Strategy**: Create projects that demonstrate production-grade skills

### Available Resources
- **Gemini Pro Subscription**: Full API access to Gemini 3 models
- **Google Antigravity IDE**: Connected to Gemini Pro account
- **Google Colab Pro**: Pro-tier cloud GPU access
- **Primary Language**: Python
- **Focus Areas**: Data Science, Machine Learning, AI

### Development Workflow
- **Claude Code** (MacOS Claude app): Primary tool for code creation and editing
- **Antigravity IDE**: Manual interaction with project directory, testing
- **GitHub**: Version control, public repository for submission

---

## 2. Hackathon Context

### Event Details
- **Name**: Gemini 3 Hackathon
- **Host**: Google DeepMind & Devpost
- **URL**: https://gemini3.devpost.com
- **Deadline**: February 9, 2026 @ 5:00pm PST
- **Prize Pool**: $100,000 total
  - Grand Prize: $50,000
  - Second Place: $20,000
  - Third Place: $10,000
  - Honorable Mentions (10): $2,000 each

### Judging Criteria
| Criterion | Weight | Description |
|-----------|--------|-------------|
| Technical Execution | 40% | Quality development, Gemini 3 leverage, code quality |
| Innovation / Wow Factor | 30% | Novelty, originality, unique solution |
| Potential Impact | 20% | Real-world usefulness, problem significance |
| Presentation / Demo | 10% | Clear demo, documentation quality |

### Target Track: "The Marathon Agent"
- Build autonomous systems for tasks spanning hours or days
- Use Thought Signatures and Thinking Levels for continuity
- Self-correct across multi-step tool calls without human supervision

### Explicitly Discouraged Projects
- Baseline RAG (1M context makes simple retrieval baseline)
- Prompt-only wrappers
- Simple vision analyzers
- Generic chatbots
- Medical/mental health diagnostic advice

### Submission Requirements
- Text description of Gemini integration (~200 words)
- Public project link (demo or hosted app)
- Public code repository URL
- ~3-minute demonstration video

---

## 3. Project Vision

### One-Liner
An autonomous agent that designs, executes, and iterates on ML experiments without human supervision, maintaining experimental reasoning across hundreds of API calls using Gemini 3's Thought Signatures.

### Core Value Proposition
Unlike AutoML tools (H2O, AutoGluon) that operate as black boxes, this agent:
- **Explains each decision** with documented reasoning
- **Generates and tests hypotheses** about why experiments succeed or fail
- **Diagnoses failures** with root cause analysis
- **Produces publication-ready reports** with narrative explanations
- **Accepts natural language constraints** rather than configuration files
- **Adapts strategy** based on accumulated results

### Input
- Dataset (CSV or Parquet)
- Target column name
- Task type (classification or regression)
- Optional: Constraints file (Markdown)

### Output
- Final trained model (best performing)
- Comprehensive experiment report (Markdown)
- MLflow experiment with full history
- Reproducible training code for each experiment
- Visualization artifacts

---

## 4. Finalized Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Execution Environment** | Subprocess | Simple, fast to implement, sufficient for hackathon |
| **State Persistence** | JSON + Pydantic | Human-readable, type-safe, easy debugging |
| **Report Format** | Markdown only | Sufficient for demo, avoids complexity |
| **MLflow Hosting** | Local | No network dependency, faster, reliable |
| **CLI Framework** | Typer | Modern, type-hint based |
| **Problem Type Detection** | User-specified | Explicit `--task` argument |
| **Metric Selection** | Gemini decides | Intelligent defaults with fallbacks |
| **Preprocessing** | Gemini decides | Flexible, learns from results |
| **Class Imbalance** | Out of scope | Stretch goal only |

### Demo Datasets (Priority Order)
1. **House Prices (Kaggle)** — Primary, regression
2. **Titanic (Kaggle)** — Secondary, classification  
3. **Credit Card Fraud** — Stretch only

---

## 5. Success Criteria

### Hackathon Success
- [ ] Working demo on House Prices dataset
- [ ] 3-minute video showing autonomous operation
- [ ] Clear differentiation from AutoML articulated
- [ ] Thought Signatures demonstrably maintaining context

### Portfolio Success
- [ ] Clean, well-documented code
- [ ] README explains value proposition clearly
- [ ] Can discuss architectural decisions in interviews
- [ ] Demonstrates ML lifecycle understanding

### Technical Success
- [ ] Agent completes 5+ iterations autonomously
- [ ] Graceful error handling
- [ ] Results reproducible via generated code
- [ ] Reports are genuinely useful

---

## 6. Constraints

### Timeline
- **Deadline**: February 9, 2026 @ 5:00pm PST
- **Available Time**: ~23 days

### Technical Constraints
- Must use Gemini 3 API as core reasoning engine
- Must demonstrate Thought Signatures
- Code must be publicly accessible (GitHub)

### Scope Boundaries
**In Scope**: Tabular data classification and regression

**Out of Scope (for hackathon)**:
- Deep learning / neural networks
- Time series forecasting
- NLP tasks
- Computer vision
- Distributed training
- Class imbalance handling (stretch only)

---

## 7. Key Documents

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Claude Code context (primary reference during coding) |
| `PROGRESS.md` | Session logs tracking progress and current status |
| `TECHNICAL_SPECIFICATION.md` | Detailed architecture, component specs |
| `IMPLEMENTATION_PLAN.md` | Phased timeline, tasks, milestones |
| `TECHNICAL_REFERENCE.md` | Code snippets, dependencies, templates |
| `CONTINUATION_GUIDE.md` | Quick orientation for new sessions |

---

*Project Brief v2.0*
*Updated: January 2026*
*Status: Planning Complete, Ready for Implementation*
