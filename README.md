# ICLR 2027 reviewers vs. AI-sounding papers

**Live page:** https://trung-dt.com/iclr-2027-ai-vs-reviews/

We matched 40,449 ICLR 2027 submissions to their scores from an AI-text detector
([Open Text Detector](https://opentextdetector.com/?collection=iclr%2F2027&sort=score)) and to
their review ratings ([ICLR 2027 Scores](https://shiqiangw.github.io/iclr-2027-scores/), which
mirrors OpenReview).

- For the 60% of papers that read least like an LLM, the detector score has no relationship with
  ratings (ρ ≈ −0.01).
- Above that, ratings fall steadily. The most AI-sounding 5% average 1.74 vs. 2.13, got 2.7× as
  many 1s, and only 1.4% cleared the top-10% cutoff.
- Across all papers the rank correlation is ρ = −0.21 (95% CI −0.22 to −0.20); the AI score
  explains about 2% of the variance in ratings. The pattern holds in all 21 primary areas.

This is a correlation, not a cause, and a detector score is not proof that a specific paper was
machine-written. Ratings are a snapshot from 6 November 2026, during the discussion period.

## Reproduce

```bash
# 1. crawl the detector's ICLR 2027 listing (2,122 pages of 20; ~10 min)
python scripts/crawl.py otd_pages 2122
# 2. review ratings
curl -sLO https://shiqiangw.github.io/iclr-2027-scores/data/papers.json
curl -sLO https://shiqiangw.github.io/iclr-2027-scores/data/site.json
# 3. analysis (numpy only)
python scripts/analyze.py otd_pages papers.json site.json data/results.json
python scripts/hero.py otd_pages papers.json data/results.json
# 4. page + social card (render src/og.html at 1200x630 to og.png)
python scripts/build.py src/template.html data/results.json index.html
```

`data/results.json` holds every number on the page. Only group-level totals are published;
per-paper scores are on the two source sites.

Not affiliated with ICLR, OpenReview, or either source site.
