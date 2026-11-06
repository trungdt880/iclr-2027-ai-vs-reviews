import json, sys, datetime
tpl, res, out = sys.argv[1:4]
SITE = 'https://iclr27.trung-dt.com/'
REPO = 'https://github.com/trungdt880/iclr-2027-ai-vs-reviews'
d = json.load(open(res)); d['built_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
js = json.dumps(d, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
html = open(tpl).read().replace('/*DATA*/', js).replace('/*SITE*/', SITE).replace('/*REPO*/', REPO)
open(out, 'w').write(html)
print('wrote', out, len(html))
