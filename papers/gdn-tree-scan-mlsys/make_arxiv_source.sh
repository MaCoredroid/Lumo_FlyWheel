#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_name="gdn-tree-scan-arxiv-source.tar.gz"
stage_dir="${paper_dir}/arxiv_build"
bundle_path="${paper_dir}/${bundle_name}"

rm -rf "${paper_dir}/arxiv_build"
rm -f "${bundle_path}"
mkdir -p "${stage_dir}"

cp "${paper_dir}/main.tex" "${stage_dir}/main.tex"
cp "${paper_dir}/ref.bib" "${stage_dir}/ref.bib"
cp "${paper_dir}/main.bbl" "${stage_dir}/main.bbl"
cp "${paper_dir}/IEEEtran.cls" "${stage_dir}/IEEEtran.cls"

(
  cd "${stage_dir}"
  COPYFILE_DISABLE=1 tar -czf "${bundle_path}" \
    main.tex ref.bib main.bbl IEEEtran.cls
)

rm -rf "${paper_dir}/arxiv_build"

printf 'Created %s\n' "${bundle_path}"
printf 'Included: main.tex ref.bib main.bbl IEEEtran.cls\n'
