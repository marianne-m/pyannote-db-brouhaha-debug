# Fake VTC DB plugin for pyannote.database

## Installation

```bash
$ git clone git@github.com:hadware/pyannte-db-vtc-debug/
$ pip install pyannote-db-vtc-debug/
```

Then, tell `pyannote.database` where to look for VTCDebug audio files.

```bash
$ cat ~/.pyannote/database.yml
Databases:
   VTCDebug: /path/to/pyanannote-db-vtc-debug/data/audio/{uri}.ogg
```

## Available protocols

There's only one protocol available for now:
- `VTCDebug.SpeakerDiarization.PoetryRecitalDiarization` : contains 3 speakers, `READER`, `AGREER` and `DISAGREER`