#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_name="gdn-tree-scan-arxiv-source.tar.gz"
stage_dir="${paper_dir}/arxiv_build/gdn-tree-scan"
bundle_path="${paper_dir}/${bundle_name}"

rm -rf "${paper_dir}/arxiv_build"
mkdir -p "${stage_dir}"

cp "${paper_dir}/main.tex" "${stage_dir}/"
cp "${paper_dir}/ref.bib" "${stage_dir}/"
cp "${paper_dir}/main.bbl" "${stage_dir}/"
cp "${paper_dir}/IEEEtran.cls" "${stage_dir}/"

(
  cd "${paper_dir}/arxiv_build"
  tar -czf "${bundle_path}" gdn-tree-scan
)

rm -rf "${paper_dir}/arxiv_build"

printf 'Created %s\n' "${bundle_path}"
printf 'Included: main.tex ref.bib main.bbl IEEEtran.cls\n'
