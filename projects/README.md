# Our NLP1 work (Rosa & Nordin)

`part1/`, `part2/`, … in the repo root are the **course's** code (pulled from
`upstream` = nlp-illc/nlp1-student). We don't edit those, so pulling the next
practical never conflicts. All of **our** work lives here, one folder per project.

```
projects/
├── README.md          ← you are here: overview, deadlines, conventions
├── AI_USAGE.md        ← running log of AI assistance (needed for the blog post)
└── project1/          ← everything for Project 1 (info, code, results, slides)
    ├── README.md          assignment summary, checklist, decisions log
    ├── assignment.md      original assignment text + course rules
    ├── 01_data_exploration.ipynb
    ├── 02_experiments.ipynb
    ├── 03_analysis.ipynb
    ├── src/               our reusable code (imported by the notebooks)
    ├── results/           CV scores as JSON  (committed: small, needed for analysis)
    ├── figures/           plots for slides / blog (committed)
    ├── models/            trained checkpoints (git-ignored, regenerate from code)
    └── slides/            slide outline + final PDF deliverable
```

## Deadlines (23:59)

| Deliverable | Due | Oral check |
|---|---|---|
| Project 1 – slides PDF + repo URL | **Fri 6 Nov** | week after |
| Project 2 | Fri 20 Nov | week after |
| Project 3 | Fri 4 Dec | week after |
| Blog post | Fri 11 Dec | – |

Submit via Canvas only. One submission counts for the whole group.

## Conventions

- Notebooks are run **from their own folder** (JupyterLab / VS Code do this by
  default). The first cell puts the repo root on `sys.path`, so both
  `from part1 import ...` (course code) and `from src import ...` (our code) work.
- Anything reused across notebooks goes in `src/`, not copy-pasted between cells.
- Every experiment writes its scores to `results/<name>.json`; the analysis
  notebook only reads those files, so it can be rerun without retraining.
- Fix random seeds (`src/config.py`) so results are reproducible.
- Log AI help in `AI_USAGE.md` as you go — much easier than reconstructing it later.

## Setup

See the root `README.md` (venv or `uv` with `requirements.txt`, Python 3.14).
