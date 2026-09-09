"""Download and verify the public pilot inputs. No archives are extracted."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    'scifact': '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165',
    'fiqa': '32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02',
}


def main():
    import os
    os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf'))
    from huggingface_hub import snapshot_download
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        path = ROOT / 'data' / f'{name}.zip'
        if not path.exists():
            temporary = path.with_suffix('.download')
            url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
            with urllib.request.urlopen(url, timeout=120) as response:
                with temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{name} archive changed; inspect before changing the pinned checksum')
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name} does not match the pinned checksum')
        print(f'{name}: verified {expected}', flush=True)
    print(snapshot_download('sentence-transformers/all-MiniLM-L6-v2',
        revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        allow_patterns=['*.json', '*.txt', 'model.safetensors']))


if __name__ == '__main__':
    main()
