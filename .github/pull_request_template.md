## What & why

<!-- What changes and why. Link the issue: Closes #N -->

## How this was verified

<!-- What you actually ran and what it showed. No claims without a command. -->

## Outputs and checkpoints

<!-- Could this change outputs for existing inputs or break saved checkpoints? Write "No" or say how. -->

---
- [ ] Every changed line traces to the issue: no unrequested refactors, renames or reformatting
- [ ] No new abstraction under 3 call sites; no new dependency (or justified above)
- [ ] `make tests` and `pre-commit run --all-files` pass locally
- [ ] Comments and docstrings explain *why*, not *what*
- [ ] Docs updated if behaviour, CLI or config changed
- [ ] Single purpose: no unrelated changes bundled in
