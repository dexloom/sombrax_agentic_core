# Contributing to SombraX Agentic Core

Thanks for your interest in contributing! This document covers the basics for
getting a change merged.

## Development setup

You need a recent stable Rust toolchain (the crate's MSRV is **1.92**).

```bash
git clone https://github.com/dexloom/sombrax_agentic_core
cd sombrax_agentic_core

cargo build --all-features
cargo test --all-features
```

## Before opening a pull request

Please make sure the following all pass locally — CI runs the same checks:

```bash
cargo fmt --check
cargo clippy --all-features -- -D warnings
cargo test --all-features
cargo doc --all-features --no-deps
```

- Keep new code consistent with the surrounding style (naming, comment density,
  idioms). Run `cargo fmt`.
- Add or update tests for behavior changes. Unit tests live alongside the code
  and in `tests/`; provider tests are mocked with `wiremock` and need no network.
- Public items should carry doc comments — the crate builds with
  `#![warn(missing_docs)]`.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## Commit and PR guidance

- Write focused commits with clear messages (Conventional Commits style —
  `feat:`, `fix:`, `docs:`, … — is appreciated but not required).
- Describe the motivation and the change in the PR body; link any related issue.
- Keep PRs scoped to one logical change where possible.

## Reporting bugs / requesting features

Open an issue with a minimal reproduction (for bugs) or a concrete use case (for
features). For provider-specific issues, include the provider id and model.

## License

By contributing, you agree that your contributions will be dual-licensed under
the [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE) licenses, consistent with
the rest of the project, without any additional terms or conditions.
