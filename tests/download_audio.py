import urllib.request
import os

url = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
out_path = os.path.join(os.path.dirname(__file__), "real_human_speech.wav")

print(f"Downloading real human speech from {url}...")
try:
    urllib.request.urlretrieve(url, out_path)
    print(f"Successfully downloaded to {out_path}")
except Exception as e:
    print(f"Error downloading: {e}")
