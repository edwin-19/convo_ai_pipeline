import nemo.collections.asr as nemo_asr
import torch
from pyctcdecode import build_ctcdecoder

class ASR_Pipe:
    def __init__(self, model_path, kenlm_path, device):
        self.load_model(model_path, device, kenlm_path)
        self.device = device
    
    def load_model(self, model_path, device, kenlm_path):
        self.asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location=device)
        self.asr_model.eval()
        
        vocab = self.asr_model.tokenizer.vocab
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=kenlm_path,
            alpha=0.6,  # Weight for KenLM (adjust based on performance)
            beta=1.5,   # Weight for word count penalty
        )
    
    def pad(self, audios, audio_lens):
        max_len = max(audio_lens)
        padded_audios = []
        for audio, audio_len in zip(audios, audio_lens):
            if audio_len < max_len:
                pad = (0, max_len - audio_len)
                audio = torch.nn.functional.pad(audio, pad)
            padded_audios.append(audio)
        padded_audios = torch.stack(padded_audios)
        return padded_audios, torch.tensor(audio_lens)

    def infer(self, audio_array):
        audio = torch.tensor(audio_array)
        padded_audios, audio_lens = self.pad([audio], [audio.shape[0]])
        logits, lengths, greedy_predictions = self.asr_model.forward(input_signal=padded_audios.to(self.device), input_signal_length=audio_lens.to(self.device))
        
        logits_2d = logits[0].detach().cpu().numpy()
        preds_text = self.decoder.decode(logits_2d)
        
        return preds_text