from pathlib import Path
import requests
import importlib

try:
    importlib.import_module('patool')
except ImportError:
    import pip
    pip.main(['install', 'patool'])
finally:
    globals()['patoolib'] = importlib.import_module('patoolib')


PWD = Path(__file__).parent.resolve()
dataDir = PWD.joinpath('Data/DataXML')
outFile = dataDir.joinpath('BrS_XML.rar')

url = 'https://zenodo.org/record/3465811/files/BrS%20XML.rar?download=1'
r = requests.get(url)

with open(outFile,'wb') as f:
    f.write(r.content)

patoolib.extract_archive(outFile, outdir=dataDir)