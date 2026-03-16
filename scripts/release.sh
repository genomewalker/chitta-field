#!/bin/bash
# Release script for chitta-field
#
# Usage:
#   ./scripts/release.sh patch       # Bug fixes (1.0.0 → 1.0.1)
#   ./scripts/release.sh minor       # New features (1.0.0 → 1.1.0)
#   ./scripts/release.sh major       # Breaking changes (1.0.0 → 2.0.0)
#   ./scripts/release.sh 1.2.0       # Explicit version
#   ./scripts/release.sh minor -y    # Skip confirmation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

get_current_version() {
    grep '^version' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'
}

bump_version() {
    local current="$1"
    local type="$2"
    IFS='.' read -r major minor patch <<< "$current"
    case "$type" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "$major.$((minor + 1)).0" ;;
        patch) echo "$major.$minor.$((patch + 1))" ;;
        *)     echo "$type" ;;
    esac
}

validate_version() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

sedi() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

BUMP_TYPE=""
AUTO_CONFIRM=false

for arg in "$@"; do
    case "$arg" in
        -y|--yes) AUTO_CONFIRM=true ;;
        *) [[ -z "$BUMP_TYPE" ]] && BUMP_TYPE="$arg" ;;
    esac
done

if [[ -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <patch|minor|major|X.Y.Z> [-y|--yes]"
    exit 1
fi

CURRENT_VERSION=$(get_current_version)
NEW_VERSION=$(bump_version "$CURRENT_VERSION" "$BUMP_TYPE")

if ! validate_version "$NEW_VERSION"; then
    echo "Error: Invalid version format: $NEW_VERSION"
    exit 1
fi

echo "=== Release: $CURRENT_VERSION → $NEW_VERSION ==="

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Uncommitted changes. Commit or stash first."
    exit 1
fi

if [[ "$AUTO_CONFIRM" != "true" ]]; then
    read -p "Proceed with release v$NEW_VERSION? [y/N] " confirm
    [[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

echo "Updating Cargo.toml..."
sedi "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
grep -q "^version = \"$NEW_VERSION\"" Cargo.toml || { echo "Cargo.toml update failed"; exit 1; }

echo "Committing version bump..."
git add Cargo.toml
git commit -m "chore: bump version to $NEW_VERSION"

echo "Creating tag v$NEW_VERSION..."
git tag "v$NEW_VERSION"

echo "Pushing to origin..."
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo "=== Release v$NEW_VERSION initiated ==="
echo "Monitor: https://github.com/genomewalker/chitta-field/actions"
echo "Release: https://github.com/genomewalker/chitta-field/releases/tag/v$NEW_VERSION"
