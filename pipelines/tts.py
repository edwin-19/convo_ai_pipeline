from neutts import NeuTTS
import os

class TTS_Pipe:
    def __init__(self, model_path, ref_path, device):
        self.load_model(model_path, device)
        
        self.ref_audio = os.path.join(ref_path, 'ref.wav')
        with open(os.path.join(ref_path, 'ref.txt'), 'r') as txtf:
            self.ref_text = txtf.readline()
    
    def load_model(self, model_path, device):
        self.tts = NeuTTS(
            backbone_repo=model_path,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device,
            language='en-us'
        )
    
    def tts_infer(self, input_text):
        ref_codes = self.tts.encode_reference(self.ref_audio)
        
        wav = self.tts.infer(input_text, ref_codes, self.ref_text)
        return wav