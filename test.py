# import sounddevice as sd
import soundfile as sf
import queue
import numpy as np
import whisper
import webrtcvad

vad = webrtcvad.Vad(2)  # 0-3, чем выше - тем чувствительнее
model = whisper.load_model("small", device="cuda", download_root="./models")

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# поток с микрофона
# stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, dtype=np.int16, blocksize=480)
# stream.start()
with sf.SoundFile("temp.wav") as f:
    blocksize = 480  # например, 30 ms при 16 kHz
    while True:
        data = f.read(blocksize, dtype='int16')
        audio_callback(data, None, None, None)
        if len(data) == 0:
            break
print("Start speaking...")
frames = []
num_silent_frames = 0
while True:
    frame: np.ndarray = q.get()
    is_speech = vad.is_speech(frame.tobytes(), sample_rate=16000)
    if is_speech:
        frames.append(frame)
    else:
        num_silent_frames += 1
        if num_silent_frames <= 10 or len(frames) == 0:  # если тишина длится более 10 кадров, считаем, что речь закончилась
            continue

        audio_data = np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0
        result = model.transcribe(audio_data, fp16=False)
        print("You said:", result["text"])
        frames = []

    num_silent_frames = 0