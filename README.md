# text2speech-en

Single speaker text-to-speech in English using CoquiTTS.

    Acoustic Model: Glow TTS
    Vocoder: Griffin-Lim
    Dataset: LJSpeech

Reference: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)

1. Download [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and update the `dataset_path` in [main.py](./main.py).
2. Set the configuration with [run.sh](./run.sh) (or/and) [main.py](./main.py)
3. Train the TTS model by executing `sh train.sh`
4. Test the TTS model by executing `sh test.sh`
5. Check the output at [output/github/out.wav](./output/github/out.wav)