# 019: Realtime Transcription

* Is there a library?

* If not, what if we fired off multiple threads to listen at different time-scales, and combine the results? For instance a short timeout could catch every 1 second of audio, and optimistically transcribe that, but then when the longterm timescale listening thread returns, a transcription will likely yield a better result, so we can override previous misses. These audio chunks can be thrown in the same queue, tagged, and we can drain short-timescale chunks off the queue if there's a more recent long-timescale chunk.


RESULT:

I've opted for recording the entire audio stream, and not doing processing before `recognize`.

There are definite efficiency gains to still be had, so I'll make a new ticket, but this works well enough for short transcription runs for now.
