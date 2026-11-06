import json, sys, time, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
KEEP = ['id','title','forum_id','number','primary_area','flags','mean_score','flagged','sentences','bib_sentences','sections','flag_model','classified']
def get(p):
    fn = f'{OUT}/p{p:05d}.json'
    if os.path.exists(fn): return 'skip'
    url = f'https://opentextdetector.com/backend/v1/browse?collection=iclr%2F2027&sort=title&page={p}'
    for a in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'iclr-correlation-study/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            items = [{k: it.get(k) for k in KEEP} for it in d['items']]
            json.dump({'page': p, 'total': d['total'], 'items': items}, open(fn, 'w'))
            return len(items)
        except Exception as e:
            time.sleep(2 * 3 ** a)
    return 'fail'
pages = int(sys.argv[2])
t = time.time()
with ThreadPoolExecutor(6) as ex:
    for i, r in enumerate(ex.map(get, range(pages))):
        if i % 200 == 0: print(i, r, round(time.time()-t), flush=True)
print('done', round(time.time()-t))
