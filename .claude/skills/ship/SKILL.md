---
name: ship
description: Ship the current work - new branch if on main, run the tests, commit, push and open a pull request for Axel to review.
argument-hint: "[commit message]"
disable-model-invocation: true
---

Ship the current work with the team's script. Never merge anything: only Axel merges.

1. Run `git status --short` and `git diff --stat HEAD`. If a file looks like a secret (`.env`, an API key, credentials), a confidential Belfius or QuantumBlack document, a large data dump, or is unrelated to the work, stop and ask the user before shipping.
2. Commit message: `$ARGUMENTS`. If it is empty, write one short imperative English sentence that describes the changes (for example "Add VIX term-structure collector"). No AI or Claude attribution.
3. Run `scripts/ship.sh "<message>"` from the repository root.
4. If the tests fail, show the failing tests and stop: nothing was committed. Offer to fix them, then ship again.
5. Otherwise, give the user the pull request link that the script printed.
