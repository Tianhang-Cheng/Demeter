"""Package prepared train/test data for transfer; uses only the standard library."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/point_transformer/soybean")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/releases/point_transformer")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or set(manifest["splits"]) != {"train", "test"}:
        raise ValueError("Expected a complete dataset with only train/test splits")
    expected = {"manifest.json"}
    groups = {}
    for split, names in manifest["splits"].items():
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Empty or duplicated {split} split")
        for name in names:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"Invalid sample name: {name}")
            group = name[:-2] if name.endswith(("_i", "_o")) else name
            if groups.setdefault(group, split) != split:
                raise ValueError(f"Plant appears in both splits: {group}")
            if manifest["samples"][name]["split"] != split:
                raise ValueError(f"Inconsistent manifest split: {name}")
            expected.add(f"{split}/{name}.pth")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Unexpected/missing dataset files: {sorted(actual ^ expected)}")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "soybean.tar.gz"
    checksum = args.output / "soybean.tar.gz.sha256"
    if archive.exists() or checksum.exists():
        raise FileExistsError("Release already exists; choose a new --output directory")
    temporary = archive.with_suffix(".gz.partial")
    # Preserve all data and provenance byte-for-byte, including the manifest.
    # Source paths in the manifest are not required by training or inference.
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        for index, relative in enumerate(sorted(expected), 1):
            path = source / relative
            if path.is_symlink() or source not in path.resolve().parents:
                raise ValueError(f"Dataset file must stay inside source: {path}")
            tar.add(path, arcname=f"data/point_transformer/soybean/{relative}", recursive=False)
            if index % 10 == 0:
                print(f"Packed {index}/{len(expected)} files", flush=True)
    digest = hashlib.sha256()
    with temporary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    temporary.replace(archive)
    checksum.write_text(f"{digest.hexdigest()}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest.hexdigest(), "files": len(expected),
                      "splits": {s: len(v) for s, v in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
