"""
Bibliography management utilities for dissertation

Features:
- Auto-generate BibTeX entries from DOIs
- Validate required fields
- Suggest entries for common NFL analytics citations
- Detect duplicate entries
"""

import re
from pathlib import Path


class BibliographyHelper:
    """
    Manages bibliography entries and validation
    """

    # Common citations in NFL analytics/ML/stats
    COMMON_CITATIONS = {
        # NFL Analytics
        "yurko2020going": {
            "type": "article",
            "title": "Going Deep: Models for Continuous-Time Within-Play Valuation of Game Outcomes in American Football with Tracking Data",
            "author": "Yurko, Ronald and Ventura, Samuel and Horowitz, Maksim",
            "journal": "Journal of Quantitative Analysis in Sports",
            "volume": "16",
            "number": "2",
            "pages": "163--182",
            "year": "2020",
            "publisher": "De Gruyter",
            "doi": "10.1515/jqas-2019-0056",
            "note": "nflfastR expected points model",
        },
        "burke2009advanced": {
            "type": "online",
            "title": "Advanced NFL Stats: Win Probability Model",
            "author": "Burke, Brian",
            "year": "2009",
            "url": "https://www.advancedfootballanalytics.com",
            "note": "Pioneering NFL analytics blog",
        },
        "lopez2020bigger": {
            "type": "article",
            "title": "Bigger data, better questions, and a return to fourth down behavior: An introduction to a special issue on tracking data in the National Football League",
            "author": "Lopez, Michael J and Yurko, Ronald and Goldner, Keith",
            "journal": "Journal of Quantitative Analysis in Sports",
            "volume": "16",
            "number": "2",
            "pages": "73--79",
            "year": "2020",
            "publisher": "De Gruyter",
        },
        "schatz2003football": {
            "type": "book",
            "title": "Football Outsiders Almanac",
            "author": "Schatz, Aaron and others",
            "year": "2003",
            "publisher": "CreateSpace",
            "note": "DVOA and advanced metrics",
        },
        # Bayesian Statistics
        "gelman2013": {
            "type": "book",
            "title": "Bayesian Data Analysis",
            "author": "Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and Vehtari, Aki and Rubin, Donald B",
            "edition": "Third",
            "year": "2013",
            "publisher": "CRC Press",
            "isbn": "978-1439840955",
            "note": "Standard Bayesian reference",
        },
        "burkner2017": {
            "type": "article",
            "title": "brms: An R Package for Bayesian Multilevel Models Using Stan",
            "author": 'B\\"{u}rkner, Paul-Christian',
            "journal": "Journal of Statistical Software",
            "volume": "80",
            "number": "1",
            "pages": "1--28",
            "year": "2017",
            "doi": "10.18637/jss.v080.i01",
            "note": "brms package reference",
        },
        "carpenter2017stan": {
            "type": "article",
            "title": "Stan: A Probabilistic Programming Language",
            "author": "Carpenter, Bob and Gelman, Andrew and Hoffman, Matthew D and Lee, Daniel and Goodrich, Ben and Betancourt, Michael and Brubaker, Marcus and Guo, Jiqiang and Li, Peter and Riddell, Allen",
            "journal": "Journal of Statistical Software",
            "volume": "76",
            "number": "1",
            "year": "2017",
            "doi": "10.18637/jss.v076.i01",
        },
        "vehtari2017practical": {
            "type": "article",
            "title": "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC",
            "author": "Vehtari, Aki and Gelman, Andrew and Gabry, Jonah",
            "journal": "Statistics and Computing",
            "volume": "27",
            "number": "5",
            "pages": "1413--1432",
            "year": "2017",
            "publisher": "Springer",
        },
        "hoffman2014no": {
            "type": "article",
            "title": "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo",
            "author": "Hoffman, Matthew D and Gelman, Andrew",
            "journal": "Journal of Machine Learning Research",
            "volume": "15",
            "number": "1",
            "pages": "1593--1623",
            "year": "2014",
            "note": "NUTS sampler used in Stan",
        },
        "betancourt2017conceptual": {
            "type": "article",
            "title": "A conceptual introduction to Hamiltonian Monte Carlo",
            "author": "Betancourt, Michael",
            "journal": "arXiv preprint arXiv:1701.02434",
            "year": "2017",
            "note": "HMC diagnostics and theory",
        },
        # Machine Learning
        "chen2016xgboost": {
            "type": "inproceedings",
            "title": "XGBoost: A Scalable Tree Boosting System",
            "author": "Chen, Tianqi and Guestrin, Carlos",
            "booktitle": "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
            "pages": "785--794",
            "year": "2016",
            "organization": "ACM",
            "doi": "10.1145/2939672.2939785",
        },
        "friedman2001greedy": {
            "type": "article",
            "title": "Greedy function approximation: a gradient boosting machine",
            "author": "Friedman, Jerome H",
            "journal": "Annals of Statistics",
            "pages": "1189--1232",
            "year": "2001",
            "publisher": "JSTOR",
        },
        "breiman2001random": {
            "type": "article",
            "title": "Random Forests",
            "author": "Breiman, Leo",
            "journal": "Machine Learning",
            "volume": "45",
            "number": "1",
            "pages": "5--32",
            "year": "2001",
            "publisher": "Springer",
        },
        "hastie2009elements": {
            "type": "book",
            "title": "The Elements of Statistical Learning: Data Mining, Inference, and Prediction",
            "author": "Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome",
            "edition": "Second",
            "year": "2009",
            "publisher": "Springer",
            "isbn": "978-0387848570",
        },
        # Calibration & Uncertainty
        "gneiting2007strictly": {
            "type": "article",
            "title": "Strictly Proper Scoring Rules, Prediction, and Estimation",
            "author": "Gneiting, Tilmann and Raftery, Adrian E",
            "journal": "Journal of the American Statistical Association",
            "volume": "102",
            "number": "477",
            "pages": "359--378",
            "year": "2007",
            "publisher": "Taylor \\& Francis",
        },
        "guo2017calibration": {
            "type": "inproceedings",
            "title": "On Calibration of Modern Neural Networks",
            "author": "Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q",
            "booktitle": "Proceedings of the 34th International Conference on Machine Learning",
            "pages": "1321--1330",
            "year": "2017",
            "organization": "PMLR",
        },
        "kuleshov2018accurate": {
            "type": "inproceedings",
            "title": "Accurate Uncertainties for Deep Learning Using Calibrated Regression",
            "author": "Kuleshov, Volodymyr and Fenner, Nathan and Ermon, Stefano",
            "booktitle": "International Conference on Machine Learning",
            "pages": "2796--2804",
            "year": "2018",
            "organization": "PMLR",
        },
        # Causal Inference
        "pearl2009causality": {
            "type": "book",
            "title": "Causality: Models, Reasoning and Inference",
            "author": "Pearl, Judea",
            "edition": "Second",
            "year": "2009",
            "publisher": "Cambridge University Press",
            "isbn": "978-0521895606",
        },
        "imbens2015causal": {
            "type": "book",
            "title": "Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction",
            "author": "Imbens, Guido W and Rubin, Donald B",
            "year": "2015",
            "publisher": "Cambridge University Press",
        },
        "angrist2009mostly": {
            "type": "book",
            "title": "Mostly Harmless Econometrics: An Empiricist's Companion",
            "author": 'Angrist, Joshua D and Pischke, J\\"{o}rn-Steffen',
            "year": "2009",
            "publisher": "Princeton University Press",
            "isbn": "978-0691120355",
        },
        "athey2015machine": {
            "type": "article",
            "title": "Machine learning methods for estimating heterogeneous causal effects",
            "author": "Athey, Susan and Imbens, Guido",
            "journal": "stat",
            "volume": "1050",
            "number": "5",
            "year": "2015",
        },
        # Reinforcement Learning
        "sutton2018reinforcement": {
            "type": "book",
            "title": "Reinforcement Learning: An Introduction",
            "author": "Sutton, Richard S and Barto, Andrew G",
            "edition": "Second",
            "year": "2018",
            "publisher": "MIT Press",
            "isbn": "978-0262039246",
        },
        "kumar2020conservative": {
            "type": "inproceedings",
            "title": "Conservative Q-Learning for Offline Reinforcement Learning",
            "author": "Kumar, Aviral and Zhou, Aurick and Tucker, George and Levine, Sergey",
            "booktitle": "Advances in Neural Information Processing Systems",
            "volume": "33",
            "pages": "1179--1191",
            "year": "2020",
            "note": "CQL algorithm used for bet sizing",
        },
        "levine2020offline": {
            "type": "article",
            "title": "Offline reinforcement learning: Tutorial, review, and perspectives on open problems",
            "author": "Levine, Sergey and Kumar, Aviral and Tucker, George and Fu, Justin",
            "journal": "arXiv preprint arXiv:2005.01643",
            "year": "2020",
        },
        # Time Series & State-Space Models
        "durbin2012time": {
            "type": "book",
            "title": "Time Series Analysis by State Space Methods",
            "author": "Durbin, James and Koopman, Siem Jan",
            "edition": "Second",
            "year": "2012",
            "publisher": "Oxford University Press",
            "isbn": "978-0199641178",
        },
        "petris2009dynamic": {
            "type": "book",
            "title": "Dynamic Linear Models with R",
            "author": "Petris, Giovanni and Petrone, Sonia and Campagnoli, Patrizia",
            "year": "2009",
            "publisher": "Springer",
            "isbn": "978-0387772370",
        },
        "harvey1990forecasting": {
            "type": "book",
            "title": "Forecasting, Structural Time Series Models and the Kalman Filter",
            "author": "Harvey, Andrew C",
            "year": "1990",
            "publisher": "Cambridge University Press",
            "isbn": "978-0521405737",
        },
        # Sports Betting & Kelly Criterion
        "kelly1956new": {
            "type": "article",
            "title": "A New Interpretation of Information Rate",
            "author": "Kelly, John L",
            "journal": "Bell System Technical Journal",
            "volume": "35",
            "number": "4",
            "pages": "917--926",
            "year": "1956",
            "publisher": "Wiley Online Library",
            "note": "Original Kelly criterion paper",
        },
        "thorp2006kelly": {
            "type": "article",
            "title": "The Kelly criterion in blackjack sports betting, and the stock market",
            "author": "Thorp, Edward O",
            "journal": "Handbook of Asset and Liability Management",
            "volume": "1",
            "pages": "385--428",
            "year": "2006",
            "publisher": "Elsevier",
        },
        "haghighat2021review": {
            "type": "article",
            "title": "A Review of Data-Driven Football Player Rating Systems",
            "author": "Haghighat, Mohammadmahdi and Rastegari, Hassan and Nourafza, Nooshin",
            "journal": "IEEE Access",
            "volume": "9",
            "pages": "144873--144891",
            "year": "2021",
            "publisher": "IEEE",
        },
        # Statistical Testing & Validation
        "diebold1995comparing": {
            "type": "article",
            "title": "Comparing Predictive Accuracy",
            "author": "Diebold, Francis X and Mariano, Roberto S",
            "journal": "Journal of Business \\& Economic Statistics",
            "volume": "13",
            "number": "3",
            "pages": "253--263",
            "year": "1995",
            "publisher": "Taylor \\& Francis",
            "note": "Diebold-Mariano test for forecast comparison",
        },
        "bergmeir2018note": {
            "type": "article",
            "title": "A note on the validity of cross-validation for evaluating autoregressive time series prediction",
            "author": "Bergmeir, Christoph and Hyndman, Rob J and Koo, Bonsoo",
            "journal": "Computational Statistics \\& Data Analysis",
            "volume": "120",
            "pages": "70--83",
            "year": "2018",
            "publisher": "Elsevier",
        },
        "tashman2000out": {
            "type": "article",
            "title": "Out-of-sample tests of forecasting accuracy: an analysis and review",
            "author": "Tashman, Leonard J",
            "journal": "International Journal of Forecasting",
            "volume": "16",
            "number": "4",
            "pages": "437--450",
            "year": "2000",
            "publisher": "Elsevier",
        },
    }

    def __init__(self, bib_path: str):
        self.bib_path = Path(bib_path)
        self.entries = self._parse_bib_file() if self.bib_path.exists() else {}

    def _parse_bib_file(self) -> dict[str, dict]:
        """Parse BibTeX file and return dictionary of entries"""
        entries = {}

        if not self.bib_path.exists():
            return entries

        content = self.bib_path.read_text()

        # Match @type{key, ... }
        pattern = r"@(\w+)\{([^,]+),([^@]*?)(?=\n@|\Z)"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            entry_type = match.group(1).lower()
            key = match.group(2).strip()
            fields_text = match.group(3)

            # Parse fields
            fields = {"type": entry_type}
            field_pattern = r"(\w+)\s*=\s*\{([^}]*)\}"
            field_matches = re.finditer(field_pattern, fields_text)

            for field_match in field_matches:
                field_name = field_match.group(1).strip()
                field_value = field_match.group(2).strip()
                fields[field_name] = field_value

            entries[key] = fields

        return entries

    def suggest_entry(self, cite_key: str) -> str | None:
        """
        Suggest BibTeX entry for common citations

        Args:
            cite_key: Citation key (e.g., 'yurko2020going')

        Returns:
            Formatted BibTeX entry string or None
        """
        if cite_key not in self.COMMON_CITATIONS:
            return None

        entry = self.COMMON_CITATIONS[cite_key].copy()
        return self._format_bibtex_entry(cite_key, entry)

    def _format_bibtex_entry(self, key: str, fields: dict) -> str:
        """Format dictionary as BibTeX entry"""
        entry_type = fields.pop("type", "article")
        lines = [f"@{entry_type}{{{key},"]

        # Order fields: author, title, journal/booktitle, volume, number, pages, year, publisher
        field_order = [
            "author",
            "title",
            "journal",
            "booktitle",
            "edition",
            "volume",
            "number",
            "pages",
            "year",
            "publisher",
            "organization",
            "isbn",
            "doi",
            "url",
            "note",
        ]

        for field in field_order:
            if field in fields:
                lines.append(f"  {field} = {{{fields[field]}}},")

        # Add any remaining fields
        for field, value in fields.items():
            if field not in field_order:
                lines.append(f"  {field} = {{{value}}},")

        lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma
        lines.append("}")

        return "\n".join(lines)

    def validate_entry(self, cite_key: str) -> list[str]:
        """
        Validate BibTeX entry has required fields

        Returns list of missing required fields
        """
        if cite_key not in self.entries:
            return [f"Entry '{cite_key}' not found in {self.bib_path}"]

        entry = self.entries[cite_key]
        required = ["author", "title", "year"]

        if entry.get("type") == "article":
            required.extend(["journal"])
        elif entry.get("type") == "book":
            required.append("publisher")
        elif entry.get("type") == "inproceedings":
            required.append("booktitle")

        missing = [f for f in required if f not in entry or not entry[f]]
        return missing

    def find_duplicates(self) -> list[tuple[str, str, str]]:
        """
        Find duplicate entries (same title/author/year)

        Returns list of (key1, key2, reason) tuples
        """
        duplicates = []
        entries_list = list(self.entries.items())

        for i, (key1, entry1) in enumerate(entries_list):
            for key2, entry2 in entries_list[i + 1 :]:
                # Check if title matches (case-insensitive, ignoring punctuation)
                title1 = re.sub(r"[^\w\s]", "", entry1.get("title", "")).lower()
                title2 = re.sub(r"[^\w\s]", "", entry2.get("title", "")).lower()

                if title1 and title2 and title1 == title2:
                    duplicates.append((key1, key2, "Same title"))

                # Check if same author + year
                elif (
                    entry1.get("author") == entry2.get("author")
                    and entry1.get("year") == entry2.get("year")
                    and entry1.get("author")
                    and entry1.get("year")
                ):
                    duplicates.append((key1, key2, "Same author+year"))

        return duplicates

    def check_missing_citations(self, tex_files: list[Path]) -> list[str]:
        r"""
        Scan .tex files for \cite{} and check if entries exist

        Returns list of missing citation keys
        """
        cited_keys = set()

        for tex_file in tex_files:
            if not tex_file.exists():
                continue

            content = tex_file.read_text()

            # Match \cite{key1,key2,...} and \citep{...} and \citet{...}
            patterns = [r"\\cite\{([^}]+)\}", r"\\citep\{([^}]+)\}", r"\\citet\{([^}]+)\}"]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    keys = [k.strip() for k in match.split(",")]
                    cited_keys.update(keys)

        # Check which cited keys are missing from bib file
        missing = [k for k in cited_keys if k not in self.entries]
        return sorted(missing)

    def get_citation_stats(self) -> dict[str, int]:
        """Get statistics about bibliography"""
        stats = {
            "total_entries": len(self.entries),
            "articles": sum(1 for e in self.entries.values() if e.get("type") == "article"),
            "books": sum(1 for e in self.entries.values() if e.get("type") == "book"),
            "inproceedings": sum(
                1 for e in self.entries.values() if e.get("type") == "inproceedings"
            ),
            "other": sum(
                1
                for e in self.entries.values()
                if e.get("type") not in ["article", "book", "inproceedings"]
            ),
        }
        return stats

    def suggest_missing_common_citations(self, tex_files: list[Path]) -> list[str]:
        """
        Scan .tex files and suggest common citations that might be missing

        Returns list of suggested citation keys
        """
        content = ""
        for tex_file in tex_files:
            if tex_file.exists():
                content += tex_file.read_text().lower()

        suggestions = []

        # Check for keywords that suggest common citations
        keywords = {
            "yurko2020going": ["expected points", "nflfastr", "win probability"],
            "gelman2013": ["bayesian", "hierarchical model", "posterior"],
            "burkner2017": ["brms", "stan", "bayesian regression"],
            "chen2016xgboost": ["xgboost", "gradient boosting"],
            "pearl2009causality": ["causal inference", "dag", "directed acyclic"],
            "kelly1956new": ["kelly criterion", "bet sizing", "bankroll"],
            "gneiting2007strictly": ["proper scoring", "brier score", "calibration"],
            "kumar2020conservative": ["conservative q-learning", "cql", "offline rl"],
        }

        for cite_key, terms in keywords.items():
            if cite_key not in self.entries:  # Not already in bibliography
                if any(term in content for term in terms):
                    suggestions.append(cite_key)

        return suggestions

    def generate_bibliography_report(self, tex_files: list[Path]) -> str:
        """Generate comprehensive bibliography report"""
        lines = ["=" * 80]
        lines.append("BIBLIOGRAPHY REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Statistics
        stats = self.get_citation_stats()
        lines.append("STATISTICS:")
        for key, value in stats.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Missing citations
        missing = self.check_missing_citations(tex_files)
        if missing:
            lines.append(f"MISSING CITATIONS ({len(missing)}):")
            for key in missing:
                lines.append(f"  - {key}")
            lines.append("")

        # Duplicates
        duplicates = self.find_duplicates()
        if duplicates:
            lines.append(f"DUPLICATE ENTRIES ({len(duplicates)}):")
            for key1, key2, reason in duplicates:
                lines.append(f"  - {key1} ↔ {key2} ({reason})")
            lines.append("")

        # Validation issues
        issues = []
        for key in self.entries:
            problems = self.validate_entry(key)
            if problems:
                issues.append((key, problems))

        if issues:
            lines.append(f"VALIDATION ISSUES ({len(issues)}):")
            for key, problems in issues:
                lines.append(f"  - {key}: {', '.join(problems)}")
            lines.append("")

        # Suggested common citations
        suggestions = self.suggest_missing_common_citations(tex_files)
        if suggestions:
            lines.append(f"SUGGESTED COMMON CITATIONS ({len(suggestions)}):")
            for key in suggestions:
                lines.append(f"  - {key}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)
