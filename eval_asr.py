import typer
from pipelines.asr import ASR_Pipe
from datasets import load_from_disk, Audio
import torch
from tqdm import tqdm
from utils import clean_text
import nemo.collections.asr.metrics.wer as wer

app = typer.Typer()

@app.command()
def main(
    model_path:str=typer.Option("./models/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo"),
    dataset_name:str=typer.Option("data/context_dialog_subset_100"),
    kenlm_path:str=typer.Option("lm/model.arpa"),
    device:str=typer.Option("cuda"),
):
    dataset = load_from_disk(dataset_name)
    dataset = dataset.cast_column("question_audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column("answer_audio", Audio(sampling_rate=16000))
    
    device = torch.device(device)
    asr_pipe = ASR_Pipe(model_path, kenlm_path, device)
    
    pred_text = [asr_pipe.infer(data['question_audio']['array']) for data in tqdm(dataset)]
    gt_text = [clean_text(d['question_text']).lower() for d in dataset]
    
    metric = wer.word_error_rate_detail(pred_text, gt_text)
    print(metric)

if __name__ == "__main__":
    app()