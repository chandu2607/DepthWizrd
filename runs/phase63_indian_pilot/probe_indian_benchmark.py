import json
import requests

bbox = '78.0,29.0,81.0,31.5'

s2_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/search?collections=sentinel-2-l2a&bbox=' + bbox + '&datetime=2024-01-01T00:00:00Z/2024-12-31T23:59:59Z&limit=5'
print('S2 search URL:', s2_url)
r = requests.get(s2_url, timeout=90)
print('S2 status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('num features:', len(data.get('features', [])))
    for f in data.get('features', [])[:5]:
        print('id:', f.get('id'))
        print('bbox:', f.get('bbox'))
        print('datetime:', f.get('properties', {}).get('datetime'))
        print('assets keys:', sorted(f.get('assets', {}).keys())[:20])
        print('---')

cop_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/search?collections=cop-dem-glo-30&bbox=' + bbox + '&datetime=2024-01-01T00:00:00Z/2024-12-31T23:59:59Z&limit=5'
print('DEM search URL:', cop_url)
r = requests.get(cop_url, timeout=90)
print('DEM status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('num dem features:', len(data.get('features', [])))
    for f in data.get('features', [])[:5]:
        print('id:', f.get('id'))
        print('bbox:', f.get('bbox'))
        print('datetime:', f.get('properties', {}).get('datetime'))
        print('assets keys:', sorted(f.get('assets', {}).keys())[:20])
        print('---')
