from datetime import datetime
import os

import whisper

models = ["tiny", "base", "small", "medium", "large"]
FILE = "absichtlich.mp3"
FILE = "msg15934454-2462.ogg"


def profile(print_args=False):
    def wrapper(func):
        def call(*args, **kwargs):
            now = datetime.now()
            _res = func(*args, **kwargs)
            print(f"{func.__name__} {[x for x in args] if print_args else ' '} took: {datetime.now() - now}")
            return _res
        return call
    return wrapper


@profile(print_args=True)
def load(model_size):
    return whisper.load_model(model_size)


@profile()
def transcribe(model, print_res=True):
    result = model.transcribe(FILE, fp16=False)
    if print_res:
        print(f"result: {result['text'].replace('. ', '.' + os.linesep)}")


def load_and_transcribe(model_size):
    transcribe(load(model_size))


# transcribe()
for m in models:
    load_and_transcribe(m)
