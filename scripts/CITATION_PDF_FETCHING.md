# Citation PDF Fetching Guide

## Problem: 403 Forbidden Errors from Sci-Hub

You're getting 403 errors because:
1. **Cloudflare Protection**: Sci-Hub uses Cloudflare which blocks automated requests
2. **Your IP May Be Blocked**: Too many requests from your IP/network
3. **Mirrors Are Down**: Sci-Hub domains change frequently

## Solution Options

### Option 1: Legal Approach (RECOMMENDED) ✅

Use the legal script that tries open access sources:

```bash
python scripts/fetch_citation_pdfs_legal.py --email your@rice.edu
```

**Sources tried:**
- ✅ **Unpaywall** - Aggregates 30M+ open access papers legally
- ✅ **arXiv** - Preprint server (most ML/stats papers)
- ✅ **PubMed Central** - Free biomedical research
- ✅ **DOAJ** - Directory of open access journals

**Advantages:**
- 100% legal
- No blocking/rate limiting issues
- Often just as effective for academic papers
- Faster (no need to bypass protections)

### Option 2: Manual Download (Most Reliable)

For papers that can't be auto-downloaded:

1. **Google Scholar** (often has free PDFs)
   ```
   https://scholar.google.com/scholar?q=PAPER_TITLE
   ```
   Look for `[PDF]` links on the right side

2. **Author Websites/ResearchGate**
   - Many authors post PDFs on their personal sites
   - ResearchGate: https://www.researchgate.net/

3. **Email Authors** (90%+ response rate!)
   ```
   Subject: Request for PDF: [Paper Title]

   Dear Dr. [Author],

   I'm a PhD student at Rice University working on my dissertation
   about NFL analytics. I came across your excellent paper "[Title]"
   and would greatly appreciate a PDF copy if you're able to share it.

   Thank you,
   [Your name]
   ```

4. **Rice University Library**
   - Your library likely has subscriptions to most journals
   - Contact library@rice.edu for access

### Option 3: Sci-Hub with VPN/Proxy

If you still want to try Sci-Hub:

1. **Use a VPN:**
   ```bash
   # Connect to VPN first, then:
   python scripts/fetch_citation_pdfs.py --delay 5.0 --limit 10
   ```

2. **Try Alternative Domains:**
   The script already tries these mirrors (updated Nov 2024):
   - https://sci-hub.se
   - https://sci-hub.st
   - https://sci-hub.ru
   - https://sci-hub.ren
   - https://sci-hub.wf
   - https://sci-hub.mksa.top

3. **Use Tor Browser:**
   ```bash
   # Install tor
   brew install tor

   # Start tor
   tor

   # Configure script to use tor proxy (requires modification)
   ```

### Option 4: Browser Extension

Install **Unpaywall browser extension** for Chrome/Firefox:
- https://unpaywall.org/products/extension

Then manually visit each DOI link and the extension will automatically find legal PDFs.

## Updated Script Features

Both scripts now include:
- ✅ Better headers to avoid detection
- ✅ Multiple mirror support (6 Sci-Hub + 3 LibGen)
- ✅ Session management with cookies
- ✅ LibGen fallback
- ✅ Graceful 403 handling
- ✅ arXiv detection

## Recommended Workflow

1. **Start with legal script:**
   ```bash
   python scripts/fetch_citation_pdfs_legal.py \
       --email your@rice.edu \
       --suggest-manual \
       --skip-existing
   ```

2. **For failures, try manual methods:**
   - Check Google Scholar
   - Email authors
   - Check Rice library

3. **Last resort - Sci-Hub with VPN:**
   ```bash
   # Connect to VPN first
   python scripts/fetch_citation_pdfs.py \
       --delay 5.0 \
       --skip-existing
   ```

## Success Rates by Method

Based on typical academic papers:

| Method | Success Rate | Legal? | Speed |
|--------|--------------|--------|-------|
| University Library | 70-80% | ✅ Yes | Fast |
| Unpaywall API | 30-40% | ✅ Yes | Fast |
| Google Scholar | 50-60% | ✅ Yes | Medium |
| Email Authors | 90%+ | ✅ Yes | Slow (24-48h) |
| arXiv/PMC | 20-30% | ✅ Yes | Fast |
| Sci-Hub (with VPN) | 80-90% | ⚠️ Gray area | Slow (if working) |

## Common Papers and Where to Find Them

### Statistics Papers (JASA, Statistical Science)
- **Best source**: Unpaywall, JSTOR (via Rice)
- Many older papers are now open access

### Machine Learning (NeurIPS, ICML, JMLR)
- **Best source**: arXiv, conference websites
- Most ML papers have free preprints

### Medical/Bio (PLOS, BMC)
- **Best source**: PubMed Central
- Many medical journals are open access

### General Science (Nature, Science)
- **Hardest to get**: Usually paywalled
- **Try**: Rice library, author preprints

## Troubleshooting

### "No module named 'bibtexparser'"
```bash
pip install bibtexparser
```

### "Invalid email address"
```bash
# Use your university email for Unpaywall
python scripts/fetch_citation_pdfs_legal.py --email yourname@rice.edu
```

### "Still getting 403 errors"
1. Use VPN and change location
2. Wait 24 hours (your IP might be temporarily blocked)
3. Try the legal script instead
4. Resort to manual methods

## Legal Notice

**Legal Script (`fetch_citation_pdfs_legal.py`):**
- ✅ 100% legal
- Uses only official APIs and open access sources
- No terms of service violations

**Sci-Hub Script (`fetch_citation_pdfs.py`):**
- ⚠️ Legal gray area
- May violate publisher terms of service
- Use for research/educational purposes only
- Check your institution's policies
- Consider it a "last resort"

## Questions?

Contact Rice University Library for help accessing papers:
- Website: https://library.rice.edu/
- Email: library@rice.edu
- Phone: (713) 348-5113
