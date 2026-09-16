# Releasing

Who can release: publishing runs in the `pypi` environment, so it is limited to maintainers of this
repository. Nothing here edits a version: the version comes from the git tag.

## Pre-release, when a sub-issue is done

A pre-release lets people try finished work without changing what plain `pip install tcri` gives
them.

1. Pick the commit on `main` whose push build is green, and note its full SHA (`git rev-parse
   origin/main`); an abbreviated SHA is rejected.
2. Preview the notes collected so far:

   ```bash
   pip install -e ".[dev]"
   towncrier build --draft --version 0.13.0a1 > notes.md
   ```

   `--draft` writes nothing and leaves the release-note files in place, so later releases still
   include them.
3. Publish the release:

   ```bash
   gh release create v0.13.0a1 --target <sha> --prerelease --notes-file notes.md
   ```

   In the web interface: Releases → Draft a new release → tag `v0.13.0a1` (create on publish),
   target that commit, paste the notes, tick **Set as a pre-release**, Publish.
4. The release workflow runs the tests, builds, checks the guard and uploads to PyPI. Install it
   with `pip install --pre tcri`.

## Release, when a tracking issue is done

1. Branch from `main`:

   ```bash
   git switch -c release/0.13.0 origin/main
   ```
2. Collect the notes into a page for the version:

   ```bash
   towncrier build --yes --version 0.13.0
   ```

   This writes `docs/release-notes/0.13.0.md`, deletes the one-line files it collected, and stages
   both. (`--yes` and `--keep` cannot be combined.)
3. Write this release's copy of what the tools store, which later releases are checked against:

   ```bash
   pytest tests/test_result_archives.py --write-archive 0.13.0
   ```

   Commit it together with the notes. The release workflow refuses to publish a release that has
   none.
4. Add an include for the new page at the top of the list in `docs/release-notes/index.md`.
5. Open a pull request with the labels `no release note` and `no issue` and the milestone of the
   release, and merge it.
6. Publish the release on that merge commit: tag `v0.13.0`, **not** a pre-release, with a body
   linking to the release notes page.
7. Close the milestone and open the next one.

## Patch release

If only fixes have landed on `main` since the last release, cut it from `main` exactly like a
release, with the patch version.

Otherwise the fix has to go out without the unreleased work on `main`:

1. Create the release branch from the tag, once per minor version:

   ```bash
   git switch -c 0.13.x v0.13.0 && git push -u origin 0.13.x
   ```
2. Open a pull request against `0.13.x` with the fix cherry-picked from `main` and `Backport of #N`
   in the description. Build the notes and the archive there in the same commit, as in steps 2 and 3
   above.
3. Publish the release from the `0.13.x` branch.
4. Open a pull request to `main` with that same commit, so the one-line note is removed there too.

A release branch cut from a tag older than the current release workflow first needs that workflow,
the build configuration and the release-note setup backported to it.

## If the release workflow fails after the release is published

- **Nothing reached PyPI:** delete the release and its tag, fix the cause, and publish the same
  version again.
- **Anything reached PyPI:** that version number is spent. Fix the cause and cut the next one.

## After any release

- PyPI shows the new version.
- The `stable` docs move to the new version. Pre-releases do not change `stable`.
- After a pre-release, the `latest` docs show the next minor's development version, because the
  version is counted from the newest tag.

## Moving or renaming things

PyPI trusts a specific repository, workflow file and environment to upload. Renaming the
repository, `.github/workflows/release.yml` or the `pypi` environment needs a new trusted publisher
on PyPI first, or the upload fails with `invalid-publisher`.
