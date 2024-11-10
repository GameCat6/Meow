import torch

device = torch.device('cpu')
torch.set_num_threads(4)
speaker = 'xenia'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
sample_rate = 48000  # 8000, 24000, 48000

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='ru',
                                     speaker='ru_v3')
model.to(device)
example_text = 'Котики'
audio = model.apply_tts(text=example_text,
                        speaker=speaker,
                        sample_rate=sample_rate)