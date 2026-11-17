#!/usr/bin/env bash
# Ship your work: new branch (if on main) -> tests -> commit -> push -> pull request.
#
# Usage:  scripts/ship.sh "what you changed"
#
# Run it again on the same branch to add commits to the open pull request.
set -euo pipefail

BASE="main"
REVIEWER="axelizsak"

die() { echo "ship: $*" >&2; exit 1; }
say() { echo "ship: $*"; }

message="${1:-}"
[ -n "$message" ] || die 'usage: scripts/ship.sh "what you changed"'

root="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$root"

# 1. Turn on the git hook that refuses pushes to main (scripts/hooks/pre-push).
git config core.hooksPath scripts/hooks

branch="$(git symbolic-ref --short -q HEAD)" || die "detached HEAD: switch to a branch first"

# 2. Is there anything to ship?
if git rev-parse -q --verify '@{u}' >/dev/null; then
    ahead="$(git rev-list --count '@{u}..HEAD')"
else
    ahead="$(git rev-list --count "origin/$BASE..HEAD" 2>/dev/null || echo 0)"
fi
git add -A
if git diff --cached --quiet && [ "$ahead" -eq 0 ]; then
    die "nothing to ship: no changes and no unpushed commits"
fi

# 3. Never ship secrets.
shipped_files="$( { git diff --cached --name-only; git diff --name-only "origin/$BASE...HEAD" 2>/dev/null || true; } | sort -u)"
bad_files="$(printf '%s\n' "$shipped_files" \
    | grep -E '(^|/)\.env($|\.)|api_key|\.pem$|(^|/)secrets\.toml$' \
    | grep -vE '(^|/)\.env\.example$' || true)"
if [ -n "$bad_files" ]; then
    printf '%s\n' "$bad_files" | while read -r f; do git reset -q -- "$f"; done
    die "refusing to ship files that look like secrets:
$bad_files"
fi
if { git diff --cached; git diff "origin/$BASE...HEAD" 2>/dev/null || true; } | grep -qE 'sk-ant-[A-Za-z0-9_-]{20,}'; then
    git reset -q
    die "refusing to ship: the changes contain what looks like an Anthropic API key"
fi

# 4. Never work on main: move the changes to a new branch.
if [ "$branch" = "$BASE" ]; then
    if [ "$ahead" -gt 0 ]; then
        say "warning: local $BASE has $ahead unpushed commit(s); they move to the new branch."
        say "         afterwards, reset $BASE with: git branch -f $BASE origin/$BASE"
    fi
    who="$(git config user.name | awk '{print tolower($1)}' | LC_ALL=C tr -cd 'a-z0-9-')"
    who="${who:-dev}"
    slug="$(printf '%s' "$message" | LC_ALL=C tr '[:upper:]' '[:lower:]' \
        | LC_ALL=C sed -E 's/[^a-z0-9]+/-/g; s/^-+//' | cut -c1-40 | sed -E 's/-+$//')"
    slug="${slug:-update}"
    new="$who/$slug"
    n=2
    while git show-ref -q --verify "refs/heads/$new" || git show-ref -q --verify "refs/remotes/origin/$new"; do
        new="$who/$slug-$n"
        n=$((n + 1))
    done
    git switch -q -c "$new"
    branch="$new"
    say "created branch $branch"
fi

# 5. Tests must pass before anything is committed.
if [ -x .venv/bin/python ]; then py=".venv/bin/python"; else py="python3"; fi
say "running tests ($py -m pytest)..."
"$py" -m pytest -q || die "tests failed, nothing was committed. Fix them and run ship again."

# 6. Commit and push.
if ! git diff --cached --quiet; then
    git commit -q -m "$message"
    say "committed: $message"
fi
git push -q -u origin "$branch"
say "pushed $branch"

# 7. Open the pull request (or print the link to open it).
remote_url="$(git remote get-url origin)"
case "$remote_url" in
    *github.com[:/]*) ;;
    *) say "origin is not on GitHub: open a pull request from $branch into $BASE by hand."; exit 0 ;;
esac
repo="$(printf '%s' "$remote_url" | sed -E 's#^.*github\.com[:/]##; s#\.git$##')"
compare_url="https://github.com/$repo/compare/$BASE...$branch?expand=1"

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
    say "GitHub CLI not available. Open the pull request here (add $REVIEWER as reviewer):"
    echo "  $compare_url"
    exit 0
fi

existing="$(gh pr list --head "$branch" --state open --json url --jq '.[0].url // empty' 2>/dev/null || true)"
if [ -n "$existing" ]; then
    say "pull request updated: $existing"
    exit 0
fi

body="$(printf 'Commits:\n%s\n\nTests: `pytest` passed locally before shipping.\n' \
    "$(git log --format='- %s' "origin/$BASE..HEAD")")"
set -- --base "$BASE" --head "$branch" --title "$message" --body "$body"
me="$(gh api user --jq .login 2>/dev/null || true)"
if [ "$me" != "$REVIEWER" ]; then
    set -- "$@" --reviewer "$REVIEWER"
fi
if url="$(gh pr create "$@")"; then
    say "pull request opened: $url"
else
    say "could not open the pull request automatically. Open it here:"
    echo "  $compare_url"
fi
