"""Build the offline blog from a readable template and verified figure artifacts."""
from pathlib import Path
import hashlib
import html
import json
import re

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    path=ROOT/'figures/publication_figure_manifest.v1.json'
    manifest=json.loads(path.read_text())
    intervention_path=ROOT/'figures/intervention_transfer_figure_manifest.v1.json'
    intervention_manifest=json.loads(intervention_path.read_text())
    for figure_manifest in (manifest,intervention_manifest):
        for name,expected in figure_manifest['input_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Figure input drift: '+name)
        for name,expected in figure_manifest['output_sha256'].items():
            # Manifest output paths are relative to the figures directory.
            p=ROOT/name if (ROOT/name).exists() else ROOT/'figures'/name
            if sha(p)!=expected:raise ValueError('Figure output drift: '+name)
    if sha(ROOT/intervention_manifest['script'])!=intervention_manifest['script_sha256']:
        raise ValueError('Intervention figure generator drift')
    article=(ROOT/'blog.template.html').read_text()
    figures={'controlled':('controlled_mechanism_v1.svg','Controlled mechanisms'),
             'natural':('natural_capacity_v1.svg','Natural retrieval and capacity'),
             'fusion':('fusion_frontier_v1.svg','Fusion quality and serving searches'),
             'intervention':('intervention_transfer_v1.svg','Intervention adaptation headroom')}
    captions={
        'controlled':'Exact constructed examples distinguish complementary evidence from an advantage that requires adapting the second probe.',
        'natural':'Development results in six correlated collection/backend settings. The capacity brackets use target labels and are not confidence intervals.',
        'fusion':'Source-selected fusion offers several quality/cost tradeoffs. The horizontal axis counts actual mean searches; generation costs remain separate.',
        'intervention':'Changing one global policy with the backend offers little additional quality on this dataset and menu. This evaluator bound does not apply to per-question routing.'}
    for key,(filename,label) in figures.items():
        raw=(ROOT/'figures'/filename).read_text()
        svg=raw[raw.index('<svg'):raw.rindex('</svg>')+6]
        ids=re.findall(r'\bid="([^"]+)"',svg)
        for ident in sorted(set(ids),key=len,reverse=True):
            svg=svg.replace('id="'+ident+'"','id="'+key+'-'+ident+'"')
            svg=re.sub(r'#'+re.escape(ident)+r'(?=[\)"\'])','#'+key+'-'+ident,svg)
        alt=(intervention_manifest['figures']['intervention'] if key=='intervention' else manifest['figures'][label])['alt_text']
        svg=svg.replace('<svg ','<svg role="img" aria-label="'+html.escape(alt,quote=True)+'" ',1)
        figure='<figure class="scientific-figure"><button type="button" data-enlarge="figure-'+key+'">Enlarge '+label.lower()+'</button><div id="figure-'+key+'">'+svg+'</div><figcaption class="caption">'+html.escape(captions[key])+'</figcaption></figure>'
        if key=='natural':figure='<details><summary>Inspect the six retrieval environments</summary><div class="body">'+figure+'</div></details>'
        article=article.replace('<!-- FIGURE:'+key+' -->',figure)
    # Source SVG files remain byte-preserved. Presentation-only whitespace is stripped.
    article='\n'.join(line.rstrip() for line in article.splitlines())+'\n'
    (ROOT/'searchprobe-blog.html').write_text(article)
    metadata={'template_sha256':sha(ROOT/'blog.template.html'),'builder_sha256':sha(Path(__file__)),
              'figure_manifest_sha256':sha(path),'intervention_figure_manifest_sha256':sha(intervention_path),
              'html_sha256':sha(ROOT/'searchprobe-blog.html'),
              'status':'Technical blog draft; no new-family confirmation or publication claim.'}
    (ROOT/'blog_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Built',ROOT/'searchprobe-blog.html')

if __name__=='__main__':main()
