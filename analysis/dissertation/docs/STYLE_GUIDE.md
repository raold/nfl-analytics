# Dissertation LaTeX Style Guide

**Document:** NFL Betting Uncertainty Quantification Dissertation
**Pages:** 752 | **Chapters:** 12 | **Tables:** 150+ | **Figures:** 50+
**Status:** Publication Ready (Jan 2026)

This guide documents the visual standards established through comprehensive chapter-by-chapter review.

---

## Table Formatting

### Standard Table Template
```latex
\begin{table}[htbp]
  \centering
  \caption[Short LOT caption]{Full descriptive caption with context and key findings.}
  \label{tab:descriptive-name}
  \begin{threeparttable}
  \scriptsize  % Use \footnotesize for moderately wide tables
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.05}
  \begin{tabularx}{\linewidth}{@{} l X X X @{}}
    \toprule
    \textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} & \textbf{Column 4} \\
    \midrule
    Data row 1 & Value & Value & Value \\
    Data row 2 & Value & Value & Value \\
    \bottomrule
  \end{tabularx}
  \begin{tablenotes}[flushleft]
    \tiny
    \item Notes and methodology explanations go here.
  \end{tablenotes}
  \end{threeparttable}
\end{table}
```

### Table Rules (booktabs)

| Rule | Usage |
|------|-------|
| `\toprule` | Top of table (thick) |
| `\midrule` | Below headers, between sections (medium) |
| `\bottomrule` | Bottom of table (thick) |
| `\cmidrule{2-4}` | Partial rule spanning columns 2-4 |

**Never use:** `\hline`, `|` (vertical rules)

### Font Sizes for Tables

| Table Width | Font Size | Use Case |
|-------------|-----------|----------|
| Narrow (3-4 cols) | `\small` or normal | Simple data |
| Medium (5-6 cols) | `\footnotesize` | Most results tables |
| Wide (7+ cols) | `\scriptsize` | Comprehensive comparisons |
| Very wide | `\tiny` + landscape | Multi-method comparisons |

### Landscape Tables
```latex
\begin{landscape}
\begin{table}[p]  % Use [p] for full-page floats
  \centering
  \caption{...}
  ...
\end{table}
\end{landscape}
```

### Table Notes with threeparttable
```latex
\begin{threeparttable}
  \begin{tabular}{...}
    Data\tnote{a} & More data\tnote{b} \\
  \end{tabular}
  \begin{tablenotes}[flushleft]
    \tiny
    \item[a] First note explanation.
    \item[b] Second note explanation.
  \end{tablenotes}
\end{threeparttable}
```

---

## Figure Formatting

### Standard Figure Template
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\textwidth]{../figures/filename.pdf}
  \caption[Short LOF caption]{Full descriptive caption explaining what the figure shows.}
  \label{fig:descriptive-name}
\end{figure}
```

### Key Rules
- **Always use** `\centering`, never `\begin{center}...\end{center}`
- **Float specifier:** `[htbp]` (never just `[h]`)
- **Path prefix:** `../figures/` from chapter directories
- **Vector formats:** PDF or TikZ preferred over PNG
- **PNG DPI:** Minimum 300 DPI for any raster images

### TikZ Diagrams
Include the standard style:
```latex
\input{../style/tikz_diagram_style.tex}
```

Standard styles available:
- `flowbox` - Rounded boxes for flowcharts
- `flowbox-wide` - Wide variant for longer text
- `arrow` - Standard directional arrows
- `note` - Callout/annotation style
- `icon` - Circular icons

### Multi-Panel Figures
```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{fig_a.pdf}
    \caption{Panel A description}
    \label{fig:main-a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \includegraphics[width=\textwidth]{fig_b.pdf}
    \caption{Panel B description}
    \label{fig:main-b}
  \end{subfigure}
  \caption{Overall figure description.}
  \label{fig:main}
\end{figure}
```

---

## Callout Boxes

### Key Discovery Box (Blue)
```latex
\begin{tcolorbox}[
  colback=blue!5!white,
  colframe=blue!50!black,
  title={\textbf{Key Discovery}},
  fonttitle=\bfseries
]
Important finding that emerged from the analysis.
\end{tcolorbox}
```

### Practical Implication Box (Green)
```latex
\begin{tcolorbox}[
  colback=green!5!white,
  colframe=green!50!black,
  title={\textbf{Practical Implication}},
  fonttitle=\bfseries
]
Actionable insight for practitioners.
\end{tcolorbox}
```

### Warning/Caution Box (Red/Orange)
```latex
\begin{tcolorbox}[
  colback=red!5!white,
  colframe=red!50!black,
  title={\textbf{Caution}},
  fonttitle=\bfseries
]
Important caveat or limitation.
\end{tcolorbox}
```

---

## Cross-References

### Standard Patterns
```latex
Chapter~\ref{chap:results}        % Chapter 9
Section~\ref{sec:methodology}     % Section 3.2
\S\ref{sec:methodology}           % Section symbol variant
Table~\ref{tab:comparison}        % Table 5.3
Figure~\ref{fig:calibration}      % Figure 7.2
Equation~\ref{eq:bayes}           % Equation 4.1
Appendix~\ref{app:proofs}         % Appendix B
```

**Key:** Always use `~` (non-breaking space) before `\ref`

### Cite Patterns
```latex
\citet{author2020}    % "Author (2020)"
\citep{author2020}    % "(Author, 2020)"
\citeauthor{author2020}  % "Author"
\citeyear{author2020}    % "2020"
```

---

## Mathematics

### Equation Environments
```latex
% Single numbered equation
\begin{equation}
  P(A|B) = \frac{P(B|A)P(A)}{P(B)}
  \label{eq:bayes}
\end{equation}

% Multi-line aligned (numbered)
\begin{align}
  \mu &= \alpha + \beta x \label{eq:mean} \\
  \sigma &= \gamma + \delta z \label{eq:sd}
\end{align}

% Multi-line aligned (unnumbered)
\begin{align*}
  y &= mx + b \\
  z &= ax^2 + bx + c
\end{align*}
```

### Notation Conventions

| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $\mu$ | Mean/location | `\mu` |
| $\sigma$ | Standard deviation | `\sigma` |
| $\beta$ | Regression coefficient | `\beta` |
| $\theta$ | Parameter vector | `\theta` |
| $\mathcal{N}$ | Normal distribution | `\mathcal{N}` |
| $\mathbb{E}$ | Expectation | `\mathbb{E}` |
| $\mathbb{P}$ | Probability | `\mathbb{P}` |

---

## Code Listings

### Stan Code
```latex
\begin{lstlisting}[language=Stan, caption={Model specification}, label={lst:stan-model}]
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  y ~ normal(mu, sigma);
}
\end{lstlisting}
```

### Python Code
```latex
\begin{lstlisting}[language=Python, caption={Feature extraction}, label={lst:python-example}]
def extract_features(game_id: str) -> dict:
    """Extract features for a single game."""
    features = {}
    features['home_epa'] = get_team_epa(game_id, 'home')
    return features
\end{lstlisting}
```

### SQL Code
```latex
\begin{lstlisting}[language=SQL, caption={Query example}, label={lst:sql-example}]
SELECT season, week, AVG(epa) as avg_epa
FROM plays
WHERE pass = 1
GROUP BY season, week;
\end{lstlisting}
```

---

## Typography

### Prevent Widows/Orphans
Already set in preamble:
```latex
\widowpenalty=10000
\clubpenalty=10000
```

### Hyphenation Control
```latex
\hyphenation{meta-learn-ing Bayes-ian}
```

### Non-Breaking Spaces
Use `~` to prevent awkward line breaks:
```latex
Figure~\ref{fig:x}
Chapter~\ref{chap:y}
Dr.~Smith
100~MB
```

### Em-Dashes vs En-Dashes
```latex
% Em-dash for parenthetical
The model---trained on 25 seasons---performs well.

% En-dash for ranges
pages 100--150
seasons 2018--2024
```

---

## Float Placement

### Specifiers
| Specifier | Meaning |
|-----------|---------|
| `h` | Here (approximately) - **avoid alone** |
| `t` | Top of page |
| `b` | Bottom of page |
| `p` | Separate float page |
| `!` | Override internal limits |

**Standard:** `[htbp]` for most floats, `[p]` for landscape/full-page

### Keep Floats in Sections
```latex
\usepackage[section]{placeins}
```

---

## File Organization

```
analysis/dissertation/
├── main/
│   ├── main.tex           # Master document
│   ├── main.pdf           # Output
│   └── references.bib     # Bibliography
├── chapter_01_intro/
│   └── chapter_01_intro.tex
├── chapter_02_lit_review/
│   ├── chapter_02_lit_review.tex
│   └── section_*.tex      # Section files
├── figures/
│   ├── out/               # Generated figures
│   └── *.tex              # TikZ source
├── tables/
│   └── *.tex              # Standalone tables
├── style/
│   ├── dissertation.cls   # Document class
│   └── tikz_diagram_style.tex
├── appendix/
│   └── appendix_*.tex
└── docs/
    └── STYLE_GUIDE.md     # This file
```

---

## Common Fixes

### Type 3 Fonts in matplotlib PDFs
Convert to outlines:
```bash
gs -o output.pdf -dNoOutputFonts -sDEVICE=pdfwrite input.pdf
```

### Overfull Hbox
Options:
1. Reword to shorter text
2. Use `\sloppy` locally
3. Add explicit hyphenation: `uncer\-tain\-ty`
4. Use `\linebreak` or `\\`

### Float Too Large for Page
1. Reduce font size (`\scriptsize`)
2. Use landscape orientation
3. Split into multiple tables
4. Use `adjustbox` package

### Hyperref Footnote Warnings
For footnotes in floats:
```latex
% Inside table
...\footnotemark{} ...

% After table
\footnotetext{The actual footnote text.}
```

Or use `threeparttable` with `\tnote{}`.

---

## Quality Checklist

Before final submission:

- [ ] `latexmk -pdf main.tex` compiles without errors
- [ ] No undefined references (`grep -i undefined main.log`)
- [ ] No Type 3 fonts (`pdffonts main.pdf | grep -i "type 3"`)
- [ ] No missing characters (`grep -i "missing character" main.log`)
- [ ] All hyperlinks work
- [ ] PDF file size reasonable (<10MB)
- [ ] TOC, LOT, LOF properly generated
- [ ] Page numbers correct in front matter (Roman) and body (Arabic)

---

*Style guide created: January 2026*
*Based on visual review of all 12 chapters (752 pages)*
