import requests
from tqdm import tqdm

for i in tqdm(range(1,41)):

    filename = f"s0{i}.zip" if i < 10 else f"s{i}.zip"
    zip_url = "https://buckeyecorpus.osu.edu/speechfiles/" + filename
    r = requests.get(zip_url, allow_redirects=True)
    open(filename, 'wb').write(r.content)