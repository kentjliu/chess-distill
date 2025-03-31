import os
import requests
import zstandard as zstd

# File containing download links
links_file = "downloads.txt"

# Directory to store downloaded and extracted files
download_dir = "lichess_chess960"
os.makedirs(download_dir, exist_ok=True)

def download_file(url, save_path):
    """Download a file from a given URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download {url}")

def decompress_zst(zst_path, output_pgn_path):
    """Decompress a .zst file to a .pgn file."""
    with open(zst_path, "rb") as compressed, open(output_pgn_path, "wb") as decompressed:
        dctx = zstd.ZstdDecompressor()
        decompressed.write(dctx.stream_reader(compressed).read())
    print(f"Decompressed: {output_pgn_path}")

# Read download links and process files
with open(links_file, "r") as f:
    for line in f:
        url = line.strip()
        if not url:
            continue
        filename = os.path.basename(url)
        zst_path = os.path.join(download_dir, filename)
        pgn_path = zst_path.replace(".zst", "")

        # Download if not already present
        if not os.path.exists(zst_path):
            download_file(url, zst_path)

        # Decompress if not already extracted
        if not os.path.exists(pgn_path):
            decompress_zst(zst_path, pgn_path)
