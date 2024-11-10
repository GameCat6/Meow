import speech_recognition as speech_r
import pyaudio
import wave
CHUNK = 1024 # определяет форму ауди сигнала
FRT = pyaudio.paInt16 # шестнадцатибитный формат задает значение амплитуды
CHAN = 1 # канал записи звука
RT = 44100 # частота 
REC_SEC = 1 #длина записи
OUTPUT = r"c:\AI2024\train_for_1211\other\sounds\output.wav"
p = pyaudio.PyAudio()
stream = p.open(format=FRT,channels=CHAN,rate=RT,input=True,frames_per_buffer=CHUNK) # открываем поток для записи
print("rec")
frames = [] # формируем выборку данных фреймов
for i in range(0, int(RT / CHUNK * REC_SEC)):
    data = stream.read(CHUNK)
    frames.append(data)
print("done")
stream.stop_stream() # останавливаем и закрываем поток 
stream.close()
p.terminate()

w = wave.open(OUTPUT, 'wb')
w.setnchannels(CHAN)
w.setsampwidth(p.get_sample_size(FRT))
w.setframerate(RT)
w.writeframes(b''.join(frames))
w.close()
LENIN = r"c:\Users\user\Downloads\sovvlast.wav "
# LENIN = r"c:\Users\user\Downloads\06-The-Wind-That-Shakes-The-Barley.wav"
sample = speech_r.WavFile(LENIN)
sample.DURATION = 4
r = speech_r.Recognizer()
# with sample as audio:
#     content = r.record(audio,duration=4)
duration = 30
with sample as audio:
    content = r.record(audio,duration=duration,offset=duration)
    r.adjust_for_ambient_noise(audio,duration=duration)
print(type(content))
# audio = speech_r.AudioData(content,)
# print(r.recognize_google(content, language="ru-RU"))
str1 = r.recognize_google(content,language="ru-RU")
print(str1,type(str1))
