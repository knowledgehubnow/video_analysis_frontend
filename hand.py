import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

def record_video_and_audio(output_filename='output_video.mp4'):
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)

    # Get the default frames per second (fps) of the camera
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get the default frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like MJPG, DIVX, etc.
    video_output = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Set up audio recording parameters
    audio_samplerate = 44100
    audio_channels = 2
    audio_filename = 'output_audio.wav'

    # Start recording audio
    audio_stream = sd.InputStream(channels=audio_channels, samplerate=audio_samplerate)
    audio_stream.start()

    audio_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_output.write(frame)

        # # Read audio data
        # audio_chunk, overflowed = audio_stream.read(audio_samplerate // fps)
        # audio_data.extend(audio_chunk)

        # Break the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop recording audio
    audio_stream.stop()

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()

    # Save audio data to the WAV file
    write(audio_filename, audio_samplerate, np.array(audio_data))

if __name__ == "__main__":
    record_video_and_audio()
