#!/bin/bash
set -euxo pipefail
git clone -o origin  --single-branch https://github.com/matplotlib/matplotlib /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard a9ba9d5d3fe9d5ac15fbdb06127f97d381148dd0
git remote remove origin
TARGET_TIMESTAMP=$(git show -s --format=%ci a9ba9d5d3fe9d5ac15fbdb06127f97d381148dd0)
git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done
git reflog expire --expire=now --all
git gc --prune=now --aggressive
AFTER_TIMESTAMP=$(date -d "$TARGET_TIMESTAMP + 1 second" '+%Y-%m-%d %H:%M:%S')
COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)
[ "$COMMIT_COUNT" -eq 0 ] || exit 1
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng
QHULL_URL="http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz"
QHULL_TAR="/tmp/qhull-2020-src-8.0.2.tgz"
QHULL_BUILD_DIR="/testbed/build"
wget -O "$QHULL_TAR" "$QHULL_URL"
mkdir -p "$QHULL_BUILD_DIR"
tar -xvzf "$QHULL_TAR" -C "$QHULL_BUILD_DIR"
python -m pip install -e .
git config --global user.email setup@swebench.config
git config --global user.name SWE-bench
git commit --allow-empty -am SWE-bench
