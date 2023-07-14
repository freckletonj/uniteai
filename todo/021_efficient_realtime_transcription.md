# 021: Efficient Realtime Transcription

As of recent commits, during a transcription window, the entire audio is saved in memory, and the whole thing is repeatedly transcribed. Inefficient.


## Options:

* Freeze transcription of earlier portions, and only re-recognize the latest portions. Perhaps a sliding window would work, but then the window must overlay with previous windows so that, eg, words aren't cut in half, and there will be some effort needed to properly align the transcribed text with the audio. This seems like a huge ergonomic improvement, but perhaps technically tough.

* Check the rms energy level of audio chunks to find the start/stop of phrases, and cut out silence

* Cut out noise? Or perhaps `whisper` was trained on enough noisy data that it already deals well with it, and this would be a significant inefficiency.
