"""Build web assets from an original website directory (Pillow and NumPy required).

Usage: python scripts/optimize_assets.py ORIGINAL_SITE
Run from the gh-pages working directory. Originals must be outside this directory.
"""
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.cwd().resolve()
SOURCE = Path(sys.argv[1]).resolve()
assert SOURCE != ROOT and ROOT not in SOURCE.parents
assert (ROOT / '.nojekyll').exists() and (SOURCE / 'index.html').exists()


def convert_image(source):
    relative = source.relative_to(SOURCE)
    target = ROOT / relative.with_suffix('.webp')
    assert not source.with_suffix('.webp').exists(), source
    with Image.open(source) as original:
        image = original.copy()
        buffer = io.BytesIO()
        # PNG diagrams keep exact decoded pixels; JPEG frames retain their dimensions.
        if source.suffix.lower() == '.png':
            image.save(buffer, 'WEBP', lossless=True, exact=True, method=6)
        else:
            image.save(buffer, 'WEBP', quality=80, method=6, icc_profile=original.info.get('icc_profile', b''))
        data = buffer.getvalue()
        with Image.open(io.BytesIO(data)) as decoded:
            assert decoded.size == original.size
            if source.suffix.lower() == '.png':
                assert np.array_equal(np.asarray(decoded.convert('RGBA')), np.asarray(original.convert('RGBA')))
        target.write_bytes(data)
    old = ROOT / relative
    assert ROOT in old.resolve().parents
    if old.exists():
        old.unlink()
    return {'path': relative.as_posix(), 'before': source.stat().st_size, 'after': len(data)}


def convert_ply(source):
    raw = source.read_bytes()
    header_end = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:header_end].decode('ascii')
    if 'format binary_little_endian' not in header:
        return None  # Tiny ASCII fixtures are already small.
    fields = []
    count = None
    in_vertex = False
    types = {'double': '<f8', 'float': '<f4', 'uchar': 'u1', 'uint': '<u4', 'int': '<i4'}
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ['element', 'vertex']:
            count = int(parts[2])
            in_vertex = True
        elif parts[0] == 'element':
            in_vertex = False
        elif in_vertex and parts[0] == 'property':
            assert len(parts) == 3 and parts[1] in types, line
            fields.append((parts[2], types[parts[1]]))
    original_type = np.dtype(fields)
    compact_type = np.dtype([(name, '<f4' if kind == '<f8' else kind) for name, kind in fields])
    vertices = np.frombuffer(raw, dtype=original_type, count=count, offset=header_end)
    compact = vertices.astype(compact_type)
    # Match the Float32BufferAttribute values created by the website's PLYLoader.
    for name, kind in fields:
        expected = vertices[name].astype('<f4') if kind == '<f8' else vertices[name]
        assert np.array_equal(compact[name], expected), (source, name)
    tail = raw[header_end + count * original_type.itemsize:]
    new_header = re.sub(r'(?m)^property double ', 'property float ', header).encode('ascii')
    result = new_header + compact.tobytes() + tail
    # Face indices and all non-vertex bytes are preserved exactly.
    assert result[len(new_header) + count * compact_type.itemsize:] == tail
    target = ROOT / source.relative_to(SOURCE)
    target.write_bytes(result)
    return {'path': source.relative_to(SOURCE).as_posix(), 'before': len(raw), 'after': len(result), 'vertices': count}


images = sorted(p for p in (SOURCE / 'static').rglob('*') if p.suffix.lower() in {'.jpg', '.png'})
targets = [p.relative_to(SOURCE).with_suffix('.webp') for p in images]
assert len(targets) == len(set(targets)), 'Image basenames collide'
with ThreadPoolExecutor(max_workers=8) as pool:
    image_results = list(pool.map(convert_image, images))
print('Images complete', flush=True)
ply_results = [result for p in (SOURCE / 'static').rglob('*.ply') if (result := convert_ply(p))]
for path in [ROOT / 'index.html', ROOT / 'static/js/index.js']:
    text = path.read_text(encoding='utf-8')
    path.write_text(re.sub(r'\.(jpg|png)\b', '.webp', text), encoding='utf-8', newline='\n')
results = image_results + ply_results
report = {'images': len(image_results), 'models': len(ply_results), 'before_bytes': sum(r['before'] for r in results), 'after_bytes': sum(r['after'] for r in results), 'files': results}
(ROOT / 'scripts/asset-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
