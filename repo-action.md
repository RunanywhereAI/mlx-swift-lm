# Repo actions: branches, pull requests, upstream syncs, releases

This is RunAnywhere's fork of `ml-explore/mlx-swift-lm`. Every merge, pull request
and release in this fork follows the rules below, whether a person or an agent does it.

## Branches

| Branch | What it is | Who writes to it |
|---|---|---|
| `ra/main` | RunAnywhere's integration and release branch, and the default branch. Every fork change lands here. | Pull requests only |
| `main` | An exact mirror of `ml-explore/mlx-swift-lm` `main`. It holds no RunAnywhere commits. | Upstream sync only |
| `<name>/<task>` | One branch per change, for example `siddhesh/pplx-v1.2-27b`. | The author |

- Never push fork work to `main`. It stays a clean mirror, so it's always clear
  which commits are ours: `git log main..ra/main`.
- Never force-push or delete `ra/main`, or any branch a release tag was cut from.
- Branch protection isn't configured yet, so nothing enforces these rules
  automatically. Follow them by hand.

## Pull requests

- Every change reaches `ra/main` through a pull request, including one-line fixes
  and documentation.
- Open one pull request per change. Don't mix a feature, a fix and an upstream
  sync in one PR.
- Commit messages are one line: a scope prefix and the concrete change, for
  example `decider: default v1.2 to one question per sequence`.
- The PR description says what changed, why, and how it was tested. Give the
  `swift build`, the tests and the real model run with its result. "Builds" alone
  isn't a test.
- Merge only after `swift build` and `swift test` pass on macOS, and the models
  the change touches have run end to end.
- Merge feature PRs with **squash** and upstream syncs with a **merge commit**
  (see below). Never rebase `ra/main`.
- The workflows inherited from upstream are scoped to `ml-explore/mlx-swift-lm`
  (`if: github.repository == ...`), so they don't run here. Until this fork has
  its own CI, the local build and test run is the gate.

## Taking upstream changes

1. Fetch upstream:
   `git remote add upstream https://github.com/ml-explore/mlx-swift-lm.git && git fetch upstream --tags`.
2. Fast-forward `main` to `upstream/main` and push it. If `main` can't
   fast-forward, something wrote to it; stop and find out what.
3. Branch `sync/<upstream-tag>` from `ra/main` and merge the upstream release tag
   into it, for example `git merge 3.32.3`.
4. Resolve conflicts in favour of upstream, except in our fork changes. Our
   commits are listed by `git log main..ra/main`.
5. Open a PR from `sync/<upstream-tag>` into `ra/main` and merge it with a merge
   commit, so the upstream history stays intact.
6. Check that the `mlx-swift` dependency still resolves to RunAnywhere's fork
   (see `RUNANYWHERE.md`).

## Releases

- Tag only a commit on `ra/main` that has been built and tested. Never tag a
  side branch.
- Tags are SwiftPM semantic versions that continue our line: the next release
  after `3.31.11` is `3.31.12`. Before tagging, check that the version doesn't
  already exist upstream (`git ls-remote --tags upstream`). A taken version
  must never be reused.
- Use annotated tags: `git tag -a 3.31.12 -m "<one line: what changed>"`.
- Never move, reuse or delete a pushed tag. SwiftPM and `Package.resolved` pin
  the version and its revision, so a moved tag breaks every consumer's resolve.
  Fix forward with the next version.
- Create a GitHub release for each tag, with notes saying what it adds.
- After tagging, bump the pin in `runanywhere-sdks` (`Package.swift` and
  `Package.resolved`) in its own PR.

## Fork-specific engineering notes

`RUNANYWHERE.md` explains why the fork exists (the `mlx-swift` runtime and Maple)
and holds the model notes.
