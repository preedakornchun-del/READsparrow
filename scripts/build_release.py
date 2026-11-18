"""Build wheel/sdist offline with preinstalled locked build tools."""
import hashlib
import os
from pathlib import Path
from setuptools import build_meta

root=Path(__file__).resolve().parents[1]
os.chdir(root)
dist=root/'dist';dist.mkdir(exist_ok=True)
assert not any(dist.glob('*.whl')) and not any(dist.glob('*.tar.gz')), 'Use a fresh source tree/dist directory.'
sdist=build_meta.build_sdist(str(dist))
wheel=build_meta.build_wheel(str(dist))
lines=[]
for name in sorted([sdist,wheel]):
    lines.append(hashlib.sha256((dist/name).read_bytes()).hexdigest()+'  '+name)
(dist/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
