"""Join Open Text Detector scores with OpenReview ratings for ICLR 2027 and emit aggregates for the report.

Primary detector metric = the site's "average score": mean per-sentence AI probability over the
main body (mean_score.body), the quantity behind sort=score. Secondary = share of body sentences
flagged at the site's default 0.1% false-positive rate (flags['body0.1']).
"""
import json, glob, sys, collections, math
import numpy as np

PAGES, PAPERS, SITE, OUT = sys.argv[1:5]
MIN_SENT = 50  # the detector site only reports a paper when it has >= 50 scored body sentences
TOP10 = 2.6667  # top-10% cutoff of paper means published on the scores site
LOW = 1.6667    # bottom ~30% of paper means
rng = np.random.default_rng(0)


def rankdata(x):
    x = np.asarray(x, float); order = np.argsort(x, kind='mergesort'); xs = x[order]
    _, start, counts = np.unique(xs, return_index=True, return_counts=True)
    ranks = np.empty(len(x)); ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return ranks


def p_from_r(r, n):  # two-sided, normal approximation (n is in the thousands)
    t = r * math.sqrt((n - 2) / max(1e-300, 1 - r * r))
    return math.erfc(abs(t) / math.sqrt(2))


pear = lambda x, y: float(np.corrcoef(x, y)[0, 1])
spear = lambda x, y: pear(rankdata(x), rankdata(y))


def boot_ci(x, y, fn, B):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    v = np.array([fn(x[i], y[i]) for i in idx])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def corr_block(x, y, B=1000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    r, s = pear(x, y), spear(x, y)
    return dict(n=int(len(x)), pearson=r, pearson_p=p_from_r(r, len(x)), pearson_ci=boot_ci(x, y, pear, B),
                spearman=s, spearman_p=p_from_r(s, len(x)), spearman_ci=boot_ci(x, y, spear, B), r2=r * r)


def summ(ys):
    ys = np.asarray(ys, float); se = float(ys.std(ddof=1) / math.sqrt(len(ys)))
    return dict(n=int(len(ys)), mean=float(ys.mean()), se=se, ci=[float(ys.mean() - 1.96 * se), float(ys.mean() + 1.96 * se)],
                top10=float((ys >= TOP10 - 1e-6).mean()), low=float((ys <= LOW + 1e-6).mean()))


# ---------- load
det = {}
files = sorted(glob.glob(f'{PAGES}/*.json'))
for f in files:
    for it in json.load(open(f))['items']:
        if it.get('forum_id'):
            det[it['forum_id']] = it
listed = json.load(open(files[0]))['total']


def share(it, mode):
    fl = (it.get('flags') or {}).get(mode)
    return fl[0] / fl[1] if fl and fl[1] >= MIN_SENT else None


rev = {}
for number, forum, title, ratings, conf, first in json.load(open(PAPERS))['rows']:
    if ratings:
        rev[forum] = dict(ratings=ratings, mean=sum(ratings) / len(ratings), conf=conf)
site = json.load(open(SITE)); snap = site['snapshots'][-1]

rows = []
for fid, r in rev.items():
    it = det.get(fid)
    if not it:
        continue
    fs = share(it, 'body0.1'); ms = (it.get('mean_score') or {}).get('body')
    if fs is None or ms is None:
        continue
    rows.append(dict(ms=ms, fs=fs, fs01=share(it, 'body0.01'), fsapp=share(it, 'nobib0.1'),
                     msapp=(it.get('mean_score') or {}).get('nobib'),
                     mean=r['mean'], ratings=r['ratings'], conf=r['conf'], area=it.get('primary_area') or 'unknown'))
N = len(rows)
ms = np.array([r['ms'] for r in rows]); fs = np.array([r['fs'] for r in rows]); mean = np.array([r['mean'] for r in rows])
nan = lambda v: np.nan if v is None else v

out = dict(n_joined=N, n_detector_listed=listed, n_detector_fetched=len(det), n_reviewed=len(rev),
           n_active=snap['active'], snapshot=snap['taken_at'], reviews=int(sum(len(r['ratings']) for r in rows)),
           min_sentences=MIN_SENT, top10_cutoff=TOP10, low_cutoff=LOW, flag_model='qwen36-35b-a3b-oct10')
out['main'] = corr_block(ms, mean, B=2000)
out['flag_main'] = corr_block(fs, mean, B=1000)
out['ms_summary'] = dict(median=float(np.median(ms)), p25=float(np.quantile(ms, .25)), p75=float(np.quantile(ms, .75)))
out['fs_summary'] = dict(median=float(np.median(fs)), exactly100=float((fs == 1).mean()), over90=float((fs >= .9).mean()))
out['rating_summary'] = dict(mean=float(mean.mean()), median=float(np.median(mean)))

# ---------- ventiles of the AI score (equal-count bins)
order = np.argsort(ms, kind='mergesort'); chunks = np.array_split(order, 20)
ven = []
for k, ix in enumerate(chunks):
    d = summ(mean[ix]); d.update(k=k + 1, lo=float(ms[ix].min()), hi=float(ms[ix].max()), med=float(np.median(ms[ix])))
    ven.append(d)
out['ventiles'] = ven

# ---------- deciles: distribution of individual review ratings
dchunks = np.array_split(order, 10)
dec = []
for k, ix in enumerate(dchunks):
    ind = [x for i in ix for x in rows[i]['ratings']]; c = collections.Counter(ind)
    d = summ(mean[ix]); d.update(k=k + 1, lo=float(ms[ix].min()), hi=float(ms[ix].max()), n_reviews=len(ind),
                                 dist={str(v): c.get(v, 0) / len(ind) for v in (1, 2, 3, 4)})
    dec.append(d)
out['deciles'] = dec
b, t = dec[0], dec[-1]
pooled = math.sqrt((mean[dchunks[0]].var(ddof=1) + mean[dchunks[-1]].var(ddof=1)) / 2)
out['decile_gap'] = dict(bottom=b['mean'], top=t['mean'], diff=t['mean'] - b['mean'], d=float((t['mean'] - b['mean']) / pooled))

# where does it start to bend? correlation inside the lower 60% vs the upper 40%
cut = np.quantile(ms, .6)
out['lower60'] = corr_block(ms[ms < cut], mean[ms < cut], B=500); out['lower60']['cut'] = float(cut)
out['upper40'] = corr_block(ms[ms >= cut], mean[ms >= cut], B=500)

# ---------- flagged-share bands (the detector site's own bands)
BANDS = [(0, .10, 'Under 10%'), (.10, .25, '10–25%'), (.25, .50, '25–50%'), (.50, .75, '50–75%'),
         (.75, .90, '75–90%'), (.90, 1.0, '90–<100%'), (1.0, 1.01, '100%')]
out['flag_bands'] = [dict(label=l, **summ(mean[(fs >= lo) & (fs < hi)])) for lo, hi, l in BANDS]

# ---------- areas
areas = collections.defaultdict(list)
for i, r in enumerate(rows):
    areas[r['area']].append(i)
ms_d, mean_d = ms.copy(), mean.copy()
for ix in areas.values():
    ms_d[ix] -= ms[ix].mean(); mean_d[ix] -= mean[ix].mean()
out['within_area'] = dict(pearson=pear(ms_d, mean_d), pearson_ci=boot_ci(ms_d, mean_d, pear, 1000))
ar = []
for a, ix in sorted(areas.items(), key=lambda kv: -len(kv[1])):
    ix = np.array(ix)
    if len(ix) < 200:
        continue
    s = spear(ms[ix], mean[ix]); z = np.arctanh(s); se = 1.06 / math.sqrt(len(ix) - 3)
    ar.append(dict(area=a, n=int(len(ix)), spearman=s, ci=[float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
                   median_ms=float(np.median(ms[ix])), mean_rating=float(mean[ix].mean())))
out['areas'] = sorted(ar, key=lambda a: a['spearman'])
out['n_area_negative'] = sum(1 for a in ar if a['ci'][1] < 0)

# ---------- robustness
rob = []
def add(label, xs, ys, note):
    blk = corr_block(xs, ys, B=500); blk.update(label=label, note=note); rob.append(blk)
add('Average AI score (body) vs mean rating', ms, mean, 'headline')
add('Average AI score (body + appendix) vs mean rating', [nan(r['msapp']) for r in rows], mean, 'appendix included, references excluded')
add('Share flagged, FPR 0.1% (site default) vs mean rating', fs, mean, 'thresholded per sentence')
add('Share flagged, FPR 0.01% vs mean rating', [nan(r['fs01']) for r in rows], mean, 'stricter threshold')
add('Share flagged, body + appendix vs mean rating', [nan(r['fsapp']) for r in rows], mean, 'appendix included')
three = np.array([len(r['ratings']) >= 3 for r in rows])
add('Average AI score vs mean rating, ≥ 3 reviews', ms[three], mean[three], 'drops thinly reviewed papers')
add('Average AI score vs lowest rating', ms, [min(r['ratings']) for r in rows], 'harshest reviewer')
add('Average AI score vs highest rating', ms, [max(r['ratings']) for r in rows], 'most generous reviewer')
add('Average AI score vs mean reviewer confidence', ms, [nan(r['conf']) for r in rows], 'not a quality signal')
out['robustness'] = rob

json.dump(out, open(OUT, 'w'), indent=1)
m = out['main']
print(f"N={N}  rho={m['spearman']:.3f} {np.round(m['spearman_ci'],3)}  r={m['pearson']:.3f} {np.round(m['pearson_ci'],3)}")
print('within-area r', round(out['within_area']['pearson'], 3), 'areas w/ neg CI', out['n_area_negative'], '/', len(ar))
print('lower60', round(out['lower60']['spearman'], 3), out['lower60']['cut'], 'upper40', round(out['upper40']['spearman'], 3))
for v in ven: print(v['k'], round(v['lo'], 3), round(v['hi'], 3), round(v['mean'], 3), round(v['top10'], 3), round(v['low'], 3))
for d in dec: print(d['k'], {k: round(x, 3) for k, x in d['dist'].items()})
for bnd in out['flag_bands']: print(bnd['label'], bnd['n'], round(bnd['mean'], 3))
for r in rob: print(f"{r['label']:<55} n={r['n']} rho={r['spearman']:.3f} r={r['pearson']:.3f}")
print(out['decile_gap'], out['ms_summary'], out['fs_summary'])
