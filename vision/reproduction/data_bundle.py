"""Package, verify or redraw the fixed numerical datasets without changing them."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / 'reproduction/data'


def expected_files():
    result = {}
    for line in (METADATA / 'files.sha256').read_text().splitlines():
        digest, path = line.split('  ', 1)
        assert path.startswith('dataset/')
        result[path.removeprefix('dataset/')] = digest
    return result


def verify(data_root, pixels=False):
    """Check the exact file set and either original bytes or decoded RGB values."""
    expected = expected_files()
    actual = {p.relative_to(data_root).as_posix() for p in data_root.rglob('*') if p.is_file()}
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    mismatch = []
    pixel_cache = {}
    pixel_refs = json.loads((METADATA / 'pixel_hashes.json').read_text()) if pixels else None
    if pixels:
        from PIL import Image
    for name in sorted(set(expected) & actual):
        path = data_root / name
        if path.is_symlink():
            mismatch.append(name)
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if pixels:
            if digest not in pixel_cache:
                with Image.open(io.BytesIO(payload)) as image:
                    pixel_cache[digest] = {
                        'rgb_sha256': hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),
                        'width': image.width, 'height': image.height,
                    }
            matches = pixel_cache[digest] == pixel_refs[expected[name]]
        else:
            matches = digest == expected[name]
        if not matches:
            mismatch.append(name)
    report = {
        'mode': 'RGB pixels' if pixels else 'PNG bytes',
        'expected_files': len(expected), 'present_files': len(actual),
        'missing_count': len(missing), 'extra_count': len(extra),
        'mismatch_count': len(mismatch),
        'missing_examples': missing[:10], 'extra_examples': extra[:10],
        'mismatch_examples': mismatch[:10],
    }
    report['passed'] = not (missing or extra or mismatch)
    return report


def pack(data_root, output):
    report = verify(data_root)
    if not report['passed']:
        raise RuntimeError(f'Refusing to package modified data: {report}')
    checksum_path = output.with_name(output.name + '.sha256')
    if output.exists() or checksum_path.exists():
        raise FileExistsError('Choose a new archive path; existing release files are never overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_files()
    entries = [(f'dataset/{name}', data_root / name, digest) for name, digest in expected.items()]
    entries += [(f'reproduction/data/{path.name}', path, None)
                for path in METADATA.iterdir() if path.is_file()]
    entries.append(('reproduction/datasets.txt', ROOT / 'reproduction/datasets.txt', None))
    # Write atomically; archive ordering, timestamps and permissions are fixed.
    with tempfile.TemporaryDirectory(prefix='.elpis-data-', dir=output.parent) as temp:
        temporary = Path(temp) / output.name
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, path, digest in sorted(entries):
                payload = path.read_bytes()
                if digest is not None and hashlib.sha256(payload).hexdigest() != digest:
                    raise RuntimeError(f'Data changed while packaging: {path}')
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # A second writer should not replace a release created while we were working.
        os.link(temporary, output)
    checksum_path.write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
                      'sha256': digest, 'dataset_files': len(expected)}, indent=2))


def regenerate(output):
    """Use the original drawing primitives with the archived filenames/splits."""
    sys.path.insert(0, str(ROOT))
    import matplotlib
    matplotlib.use('Agg')
    from dataMaker_commonFunc import (DOT_POSITIONS, ZHENG_POSITIONS,
                                     EU_tally_mark_POSITIONS, MARK_NAME_SPACE,
                                     plot_a_scatter, plot_lines, zheng_0)
    from matplotlib import rc_context
    datasets = json.loads((METADATA / 'splits.json').read_text())['datasets']
    inverse_markers = {value: key for key, value in MARK_NAME_SPACE.items()}
    output.mkdir(parents=True, exist_ok=False)
    rendered = {}
    count = 0
    with tempfile.TemporaryDirectory(prefix='elpis-prototypes-') as temporary, rc_context({'savefig.dpi': 100, 'figure.dpi': 100}):
        cache = Path(temporary)

        def image_for(kind, label, marker, color):
            marker = inverse_markers.get(marker, marker)
            key = (kind, label, marker if kind == 'dots' else 'lines', color)
            if key not in rendered:
                path = cache / f'{len(rendered)}.png'
                if kind == 'dots':
                    plot_a_scatter(DOT_POSITIONS[label], str(path), marker=marker,
                                   color=color, is_fill=label != 0, size=18, show_axes=True)
                else:
                    positions = ZHENG_POSITIONS if kind == 'zheng' else EU_tally_mark_POSITIONS
                    # The archived tally zero is a short horizontal mark in both
                    # families, unlike the later default zero-coordinate tables.
                    lines = zheng_0(scalar=0.5) if label == 0 else positions[label]
                    plot_lines(lines, str(path), color=color,
                               line_width=2.0 if kind == 'zheng' else 1.5)
                rendered[key] = path
            return rendered[key]

        for name, recipe in datasets.items():
            folder = output / name
            if 'images' in recipe:
                folder.mkdir()
                for filename in recipe['images']:
                    label, marker, color = Path(filename).stem.split('-', 2)
                    shutil.copyfile(image_for(recipe['kind'], int(label), marker, color), folder / filename)
                    count += 1
            else:
                for split, samples in recipe['splits'].items():
                    for sample, filenames in samples.items():
                        _, _, marker, color = sample.split('-', 3)
                        destination = folder / split / sample
                        destination.mkdir(parents=True)
                        for filename in filenames:
                            _, label = Path(filename).stem.split('-', 1)
                            shutil.copyfile(image_for(recipe['kind'], int(label), marker, color), destination / filename)
                            count += 1
    print(json.dumps({'output': str(output), 'images': count, 'rendered_prototypes': len(rendered)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('verify', help='Verify the authoritative file list and hashes.')
    check.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    check.add_argument('--pixels', action='store_true', help='Compare decoded RGB pixels instead of PNG bytes.')
    bundle = commands.add_parser('pack', help='Create a ZIP release and a SHA-256 sidecar.')
    bundle.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    bundle.add_argument('--output', type=Path, required=True)
    render = commands.add_parser('regenerate', help='Redraw fixed data using archived split membership.')
    render.add_argument('--output', type=Path, required=True, help='A new directory; existing directories are rejected.')
    args = parser.parse_args()
    if args.command == 'verify':
        report = verify(args.data_root.resolve(), args.pixels)
        print(json.dumps(report, indent=2))
        if not report['passed']:
            raise SystemExit(1)
    elif args.command == 'pack':
        pack(args.data_root.resolve(), args.output.resolve())
    else:
        regenerate(args.output.resolve())


if __name__ == '__main__':
    main()
