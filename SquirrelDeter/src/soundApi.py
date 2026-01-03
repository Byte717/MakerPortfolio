import numpy as np
import sounddevice as sd



def playSound(frequency, duration):
    # frequncy in hertz, any number
    # duration in miliseconds
    SAMPLE_RATE = 44100 # hz 
    duration_sec = duration / 1000.0
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    sd.play(wave, SAMPLE_RATE)
    sd.wait()