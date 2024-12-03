import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import IPython.display as display

#load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

#load any audio file of your choice
speech, rate = librosa.load("batman1.wav",sr=16000)


display.Audio("batman1.wav", autoplay=True)

input_values = tokenizer(speech, return_tensors = 'pt').input_values

input_values

logits = model(input_values).logits

#Store predicted id's
predicted_ids = torch.argmax(logits, dim =-1)

#decode the audio to generate text
transcriptions = tokenizer.decode(predicted_ids[0])

print(transcriptions)