from pathlib import Path
import os
import datetime

DATADIR = os.getenv('DATADIR')
if DATADIR is None:
    DATADIR = Path(__file__).resolve().parent / 'Data'
else:
    DATADIR = Path(DATADIR) / 'ManagedData' / 'Data'

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")