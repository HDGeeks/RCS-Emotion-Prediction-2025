#!/bin/bash
# render.sh — compile a XeLaTeX file to PDF
# Usage: ./render.sh <filename.tex>

set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Usage: ./render.sh <filename.tex>"
  exit 1
fi

TEX="${1%.tex}"

echo "=== Removing existing PDF ==="
rm -f "${TEX}.pdf"

echo "=== Pass 1: XeLaTeX ==="
xelatex -interaction=nonstopmode "${TEX}.tex"

echo "=== BibTeX ==="
bibtex "${TEX}"

echo "=== Pass 2: XeLaTeX ==="
xelatex -interaction=nonstopmode "${TEX}.tex"

echo "=== Pass 3: XeLaTeX ==="
xelatex -interaction=nonstopmode "${TEX}.tex"

echo "=== Cleaning temp files ==="
rm -f "${TEX}.aux" "${TEX}.bbl" "${TEX}.blg" "${TEX}.log" "${TEX}.out" "${TEX}.toc" missfont.log

echo "=== Done: ${TEX}.pdf ==="
