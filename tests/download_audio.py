import os
import urllib.request

url = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
out_path = os.path.join(os.path.dirname(__file__), "real_human_speech.wav")

if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
    print(f"Sample verified: {out_path}")
else:
    print(f"Downloading real speech sample from {url}...")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        print(f"Successfully downloaded to {out_path}")
    except Exception as e:
        print(f"Error downloading: {e}")