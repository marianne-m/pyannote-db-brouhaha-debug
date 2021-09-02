# The MIT License (MIT)

# Copyright (c) 2021 COML

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# AUTHORS
# Hadrien TITEUX
import json
import logging
import random
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.io.wavfile as wavefile
from pyannote.core import Segment, Annotation
from sox import Transformer
from voxpopuli import Voice

FILES_COUNTS = {
    "train": 10,
    "dev": 5,
    "test": 5
}

AUDIO_MEAN_DURATION = 3 * 60  # in seconds
SILENCE_MEAN_DURATION = 5  # seconds
LONG_UTT_TEXT_PATH = Path(__file__).parent / Path("data/utterances.txt")
SAMPLING_RATE = 16000

BASE_TRANSFORM = Transformer(). \
    rate(SAMPLING_RATE)
BASE_TRANSFORM.set_output_format(rate=SAMPLING_RATE,
                                 encoding="signed-integer",
                                 bits=16)

REVERB_TRANSFORM = Transformer(). \
    rate(SAMPLING_RATE) \
    .reverb().vol(gain=0.9)
REVERB_TRANSFORM.set_output_format(rate=SAMPLING_RATE,
                                   encoding="signed-integer",
                                   bits=16)
CHORUS_TRANSFORM = Transformer(). \
    rate(SAMPLING_RATE) \
    .chorus().vol(gain=0.9)
CHORUS_TRANSFORM.set_output_format(rate=SAMPLING_RATE,
                                   encoding="signed-integer",
                                   bits=16)

OGG_TRANSFORM = Transformer()
OGG_TRANSFORM.set_output_format(file_type="ogg")

@dataclass
class Utterance:
    array: np.ndarray
    rate: float
    start: float  # in seconds

    @property
    def segment(self):
        return Segment(self.start, self.start + (len(self.array) / self.rate))


@dataclass
class Silence:
    duration: float
    start: float  # in seconds

    @property
    def segment(self):
        return Segment(self.start, self.start + self.duration)

    @property
    def array(self) -> np.array:
        return np.zeros(int(SAMPLING_RATE * self.duration) + 1, dtype=np.int16)


@dataclass
class Speaker:
    speaker_class: str
    utt_text: List[str]
    transformer: Optional[Transformer]
    voice: Voice = None

    def __post_init__(self):
        self.voice = Voice(speed=random.randint(100, 160),
                           pitch=random.randint(20, 80),
                           voice_id=random.randint(1, 7))

    def gen_utt(self, start: float):
        audio_wav = self.voice.to_audio(random.choice(self.utt_text))
        rate, array = wavefile.read(BytesIO(audio_wav))
        array = self.transformer.build_array(input_array=array,
                                             sample_rate_in=rate,
                                             )

        return Utterance(array, SAMPLING_RATE, start=start)


@dataclass
class Track:
    speaker: Speaker
    duration: float  # seconds
    utt_list: List[Utterance] = field(default_factory=list)
    track: List[Union[Utterance, Silence]] = field(default_factory=list)

    def render(self):
        total_dur = 0
        flag = False
        while total_dur < self.duration:
            if flag:
                new_utt = self.speaker.gen_utt(total_dur)
            else:
                new_utt = Silence(abs(random.gauss(SILENCE_MEAN_DURATION, 3)), total_dur)
            self.utt_list.append(new_utt)
            total_dur += new_utt.segment.duration
            flag = not flag

        arrays = [utt.array for utt in self.utt_list]
        return np.concatenate(arrays)


@dataclass
class DebugFile:
    tracks: List[Track]
    file_id: str
    audio: np.array = None
    annot: Annotation = None

    def render(self):
        arrays = [t.render() for t in self.tracks]
        min_len = len(min(arrays, key=lambda x: len(x)))
        arrays = [arr[:min_len] for arr in arrays]
        self.audio = (np.vstack(arrays) / 3).sum(axis=0).astype(np.int16)
        self.annot = Annotation(uri=self.file_id)
        for track in self.tracks:
            for utt in track.utt_list:
                if isinstance(utt, Silence):
                    continue
                self.annot[utt.segment] = track.speaker.speaker_class

    def save(self, audio_folder: Path, annot_folder: Path):
        annot_folder.mkdir(parents=True, exist_ok=True)
        OGG_TRANSFORM.build_file(input_array=self.audio,
                                 sample_rate_in=SAMPLING_RATE,
                                 output_filepath=str(audio_folder / Path(f"{self.file_id}.ogg")))
        wavefile.write(audio_folder / Path(f"{self.file_id}.wav"), SAMPLING_RATE, self.audio)
        with open(annot_folder / Path(f"{self.file_id}.json"), "w") as json_file:
            json.dump(self.annot.for_json(), json_file)


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("sox").setLevel(logging.CRITICAL)

    with open(LONG_UTT_TEXT_PATH) as utt_file:
        long_utterances = utt_file.read().split("\n")

    short_agree = ["ah oui", "ok", "je vois", "certes", "sans aucun doute", "bienentendu",
                   "d'accord", "très bien", "de même", "mais oui!", "évidemment"]

    short_disagree = ["non", "je ne pense pas", "ben non", "mais non!", "absolument pas",
                      "bof", "mouai", "c'est pas trop ça", "moyen"]

    data_folder = Path(__file__).parent / Path("data")
    file_id_counter = 0
    for subset, files_count in FILES_COUNTS.items():
        for _ in range(files_count):
            spkr_a = Speaker("READER", long_utterances, transformer=BASE_TRANSFORM)
            spkr_b = Speaker("AGREER", short_agree, transformer=REVERB_TRANSFORM)
            spkr_c = Speaker("DISAGREER", short_disagree, transformer=CHORUS_TRANSFORM)
            duration = random.gauss(AUDIO_MEAN_DURATION, 10)
            file = DebugFile(tracks=[Track(speaker, duration)
                                     for speaker
                                     in (spkr_a, spkr_b, spkr_c)],
                             file_id=f"DEBUG{file_id_counter}")
            logging.info(f"Rendering file {file.file_id}")
            file.render()
            logging.info(f"Saving file {file.file_id}")
            file.save(data_folder / Path("audio"),
                      data_folder / Path("annotations") / Path(subset))
            file_id_counter += 1
