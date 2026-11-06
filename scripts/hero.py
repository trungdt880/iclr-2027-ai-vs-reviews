"""Extra share-friendly numbers: compare the most AI-like 5% / 10% with the least AI-like 60%."""
import json, glob, sys, collections
import numpy as np
PAGES, PAPERS, RES = sys.argv[1:4]
det = {it['forum_id']: it for f in glob.glob(f'{PAGES}/*.json') for it in json.load(open(f))['items'] if it.get('forum_id')}
rows = []
for number, forum, title, ratings, conf, first in json.load(open(PAPERS))['rows']:
    it = det.get(forum)
    if not ratings or not it: continue
    fl = (it.get('flags') or {}).get('body0.1'); ms = (it.get('mean_score') or {}).get('body')
    if not fl or fl[1] < 50 or ms is None: continue
    rows.append((ms, ratings))
rows.sort(key=lambda r: r[0]); n = len(rows)
def grp(a, b):
    g = rows[int(a * n):int(b * n)]
    ind = [x for _, rs in g for x in rs]; c = collections.Counter(ind)
    means = np.array([sum(rs) / len(rs) for _, rs in g])
    return dict(papers=len(g), reviews=len(ind), ones=c[1] / len(ind), threes_up=(c[3] + c[4]) / len(ind),
                mean=float(means.mean()), top10=float((means >= 2.6667 - 1e-6).mean()), low=float((means <= 1.6667 + 1e-6).mean()),
                any_one=float(np.mean([1 in rs for _, rs in g])), all_one=float(np.mean([all(x == 1 for x in rs) for _, rs in g])),
                score_lo=float(g[0][0]), score_hi=float(g[-1][0]))
H = dict(bottom60=grp(0, .6), top10=grp(.9, 1), top5=grp(.95, 1), top1=grp(.99, 1), all=grp(0, 1))
res = json.load(open(RES)); res['hero'] = H; json.dump(res, open(RES, 'w'), indent=1)
for k, v in H.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
