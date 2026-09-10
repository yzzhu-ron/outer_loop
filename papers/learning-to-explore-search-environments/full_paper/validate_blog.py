"""Check the self-contained article, figure provenance, links, and displayed data."""
from html.parser import HTMLParser
from pathlib import Path
import collections
import hashlib
import json
from urllib.parse import urlsplit, unquote

ROOT=Path(__file__).resolve().parent
class Document(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.assets=[];self.text=[];self.ignore=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        for key in ('href','xlink:href'):
            if key in attrs:self.links.append(attrs[key])
        if tag in ('script','style'):self.ignore+=1
        if tag=='script' and attrs.get('src'):self.assets.append(attrs['src'])
        if tag=='link' and attrs.get('rel')=='stylesheet':self.assets.append(attrs.get('href'))
        if tag in ('img','image') and attrs.get('src',attrs.get('href','')).startswith(('http:','https:')):self.assets.append(attrs)
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.ignore-=1
    def handle_data(self,data):
        if not self.ignore:self.text.append(data)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate():
    article=ROOT/'searchprobe-blog.html';doc=Document();doc.feed(article.read_text())
    duplicates=[k for k,v in collections.Counter(doc.ids).items() if v>1]
    if duplicates:raise ValueError('Duplicate IDs: '+repr(duplicates[:10]))
    if doc.assets:raise ValueError('External page assets: '+repr(doc.assets))
    for link in doc.links:
        parts=urlsplit(link)
        if parts.scheme:continue
        if not parts.path:
            if parts.fragment and unquote(parts.fragment) not in doc.ids:raise ValueError('Broken anchor '+link)
        elif not (ROOT/unquote(parts.path)).exists():raise ValueError('Missing local link '+link)
    provenance=json.loads((ROOT/'blog_provenance.json').read_text())
    if provenance['html_sha256']!=sha(article):raise ValueError('Article hash drift')
    for key,path in [('template_sha256',ROOT/'blog.template.html'),('builder_sha256',ROOT/'build_blog.py'),('figure_manifest_sha256',ROOT/'figures/publication_figure_manifest.v1.json'),('intervention_figure_manifest_sha256',ROOT/'figures/intervention_transfer_figure_manifest.v1.json')]:
        if provenance[key]!=sha(path):raise ValueError(key+' drift')
    visible=' '.join(doc.text)
    runs=json.loads((ROOT/'natural/results/runs.json').read_text())
    baseline=json.loads((ROOT/'natural/results/baselines.json').read_text())
    expected=[]
    for name in ('original','rrf_all_actions','source_query_bucket'):
        rows=[r['ndcg'] for r in baseline if r['baseline']==name];expected.append(sum(rows)/len(rows))
    rows=[r['ndcg'] for r in runs if r['condition']=='standard' and r['method']=='lookahead2' and r['budget']==64];expected.append(sum(rows)/len(rows))
    for value in expected:
        if f'{value:.4f}' not in visible:raise ValueError('Displayed primary metric missing')
    intervention=json.loads((ROOT/'intervention_transfer/results.v1.json').read_text())
    for method in ('within','source_fixed_train_cal','rrf_all'):
        if f"{intervention['macro'][method]:.4f}" not in visible:
            raise ValueError('Displayed intervention metric missing: '+method)
    capacity=json.loads((ROOT/'theory/posthoc_intervention_headroom.v1.json').read_text())
    for policy_class,value in capacity['macro']['values'].items():
        if f'{value:.4f}' not in visible:
            raise ValueError('Displayed posthoc capacity missing: '+policy_class)
    return {'unique_ids':len(doc.ids),'links':len(doc.links),'external_assets':len(doc.assets),'provenance':'verified','primary_displayed_metrics':'verified','approximate_word_count':len(visible.split())}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
