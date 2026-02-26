#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <version> [--push] [--publish]"
    echo ""
    echo "  <version>    Semantic version (e.g. 0.2.0)"
    echo "  --push       Push the tag (and commit) to origin"
    echo "  --publish    Publish the crate to crates.io"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

VERSION="$1"
shift

# Validate semver format.
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "Error: '$VERSION' is not a valid semantic version"
    exit 1
fi

PUSH=false
PUBLISH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH=true ;;
        --publish) PUBLISH=true ;;
        *) echo "Unknown flag: $1"; usage ;;
    esac
    shift
done

# Ensure working tree is clean.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is not clean. Commit or stash changes first."
    exit 1
fi

TAG="v${VERSION}"

# Ensure tag doesn't already exist.
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag '$TAG' already exists"
    exit 1
fi

# Bump version in Cargo.toml.
sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" Cargo.toml

# Update Cargo.lock.
cargo update --workspace --quiet

# Commit and tag.
git add Cargo.toml Cargo.lock
git commit -m "Release ${TAG}"
git tag "$TAG"

echo "Created commit and tag ${TAG}"

if $PUSH; then
    git push origin main "$TAG"
    echo "Pushed to origin"
fi

if $PUBLISH; then
    cargo publish
    echo "Published cf1-rs ${VERSION} to crates.io"
fi
