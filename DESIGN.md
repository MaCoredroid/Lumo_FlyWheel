# Design

## Theme

Research-lab mini-paper for GitHub Pages. The surface should feel like a clean technical blog preview, not a dashboard and not a generic landing page. Previous Pages artifacts used single-file HTML, inline CSS, Chart.js, serif/sans/mono typography, and evidence-heavy narrative sections. Preserve the self-contained static-page model and improve scanability.

## Color

Use a restrained but distinctive light research palette instead of repeating the dark Vol. II editorial treatment. Prefer neutral off-white or very light gray backgrounds, high-contrast ink, one strong technical accent, and small secondary colors for status roles:

- ink: near-black for body and headings
- surface: white or near-white for figures and tables
- rule: quiet gray for dividers
- accent: saturated blue-green or deep cyan for proof/kernel emphasis
- caution: amber for pending gates
- pass: green for passed gates
- fail: red for rejected paths

Avoid gradient text, glass effects, decorative orbs, and color used only for atmosphere.

## Typography

Use a readable body family plus a compact mono for code, evidence labels, and tables. Headings should be large enough to create a paper-like hierarchy but not oversized. Body copy should stay around 65-75 characters per line. Tables, ledgers, and figure captions should use tabular numerals where possible.

## Layout

Single-page static document with a narrow reading column and wider full-bleed figure bands where useful. Recommended sections:

- masthead and abstract
- one-screen thesis diagram
- kernel design
- attention and wiring chase-down
- lossless evidence ladder
- measurement infrastructure
- speed attempts and current overhead frontier
- paper trail and next gates

Use cards only for genuine repeated evidence items or tables. Avoid nested cards and repeated decorative section markers.

## Components

- Evidence ladder: compact rows with status, claim, source, and gate state.
- Figure bands: diagrams for GDN scan/replay/WY concepts and measurement pipeline.
- Tables: deployment screen, trust-ledger excerpts, and attempts matrix.
- Callouts: "stands", "remeasure", and "pending" states with clear language.
- Navigation: sticky or top-of-page links to prior volumes and repository sources.

## Motion

Motion should be minimal and non-essential: subtle figure reveals or no motion. Respect `prefers-reduced-motion: reduce`. Never hide content by default while waiting for scroll-triggered JavaScript.
