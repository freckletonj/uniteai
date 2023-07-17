'''

NOTES:

* Bark was trained on 14 sec chunks.

* When it generates, it chooses a voice that's statistically likely for the audio.

* You can generate a voice with one prompt, and save it as history to be the voice of future generations.



pip install git+https://github.com/suno-ai/bark.git

'''

import os
os.environ["SUNO_USE_SMALL_MODELS"] = "True"


from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import sounddevice as sd
import nltk
# Ensure that the necessary tokenizers are downloaded
nltk.download('punkt')

# download and load all models
preload_models()

text_prompt = '''
Kneel before me puny minions. I am the god Poseidon, and those who defy me feel my mighty wrath. Tremble, as my deep voice resonates against your heart.
'''.strip()

history = generate_audio(
    text_prompt,
    text_temp=0.5,
    waveform_temp=0.5,
    output_full=True
)

sd.play(history[1], samplerate=SAMPLE_RATE)
sd.wait()


# generate audio from text
text_prompt = """
Greetings, valiant Jeremy. You find yourself in a grand entrance, enveloped by a sense of calm despite the surrounding uncertainty. A large, ornate mirror reflects your face and beyond it rest a lush, intriguing carpet spread across the floor. You are not alone, for a character, the grounds keeper, is also present in the room.
"""

def pair_strings(strings):
    return [strings[i] + " " + strings[i + 1] for i in range(0, len(strings), 2)]

# This is how you can split it up into sentences
sentences = nltk.sent_tokenize(text_prompt)

sentences = pair_strings(sentences)

for sentence in sentences:
    audio_array = generate_audio(
        sentence.strip(),
        history_prompt=history[0],
        text_temp=0.6,
        waveform_temp=0.8,
    )

    # # save audio to disk
    # write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)

    sd.wait() # wait for previous gen
    sd.play(audio_array, samplerate=SAMPLE_RATE)
