#!/usr/bin/env python3
"""Configure existing immutable ECR image tags or digests; never builds images."""
import argparse, pathlib, re
p=argparse.ArgumentParser()
for role in ['api','camera-worker','anpr-worker','frs-worker']:
    p.add_argument('--'+role, required=True)
a=p.parse_args(); root=pathlib.Path(__file__).resolve().parents[1]
refs=vars(a)
for role,ref in refs.items():
    if 'REPLACE' in ref or '<' in ref or not re.fullmatch(r'[\w./:@-]+',ref):
        p.error('Invalid image reference for '+role)
    if not ('@sha256:' in ref or (':' in ref.rsplit('/',1)[-1] and not ref.endswith(':latest'))):
        p.error('Use a version tag or sha256 digest for '+role)
for path in [*root.glob('*/deployment.yaml'),root/'jobs/migrate.yaml',root/'jobs/minio-init.yaml',root/'optional/onboarding.yaml']:
    s=path.read_text()
    for role in ['camera-worker','anpr-worker','frs-worker','api']:
        ref=refs[role.replace('-','_')]
        # Only application image fields; safe to rerun for subsequent releases.
        expected={'camera-worker':'camera-worker','anpr-worker':'anpr-worker','frs-worker':'frs-worker','api':'api'}[role]
        belongs=(path.parent.name==expected) or (role=='api' and path.parent.name in ['jobs','optional'])
        if belongs:s=re.sub(r'(?m)^(\s*image:).+$',lambda m:m.group(1)+' '+ref,s)
    path.write_text(s)
print('Application image references updated. No images built or pushed.')
