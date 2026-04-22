import typer
from datasets import load_from_disk, Audio
import torch
import os

from tqdm import tqdm
import soundfile as sf
import json

from pipelines.llm import LLM_Pipe
from pipelines.asr import ASR_Pipe
from pipelines.tts import TTS_Pipe

app = typer.Typer(pretty_exceptions_enable=False)

@torch.inference_mode()
@app.command()
def main(
    model_path:str=typer.Option("./models/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo"),
    dataset_name:str=typer.Option("data/context_dialog_subset_100"),
    kenlm_path:str=typer.Option("lm/model.arpa"),
    llm_path:str=typer.Option("./models/LFM2.5-350M"),
    tts_path:str=typer.Option("./models/neutts-nano"),
    device:str=typer.Option("cpu"),
    ref_path:str=typer.Option('ref'),
    output_path:str=typer.Option('output')
):
    dataset = load_from_disk(dataset_name)
    dataset = dataset.cast_column("question_audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column("answer_audio", Audio(sampling_rate=16000))
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    device = torch.device(device)
    asr_pipe = ASR_Pipe(model_path, kenlm_path, device)
    llm_pipe = LLM_Pipe(llm_path, device)
    tts_pipe = TTS_Pipe(tts_path, ref_path, device)
    
    output_data = []
    for index, data in tqdm(enumerate(dataset)):
        pred_text = asr_pipe.infer(data['question_audio']['array'])
        prompt = f"""
        You are now speaking as the person described in the context below. 
        Your goal is to answer the question using only the provided context, but in your unique voice, vocabulary, and perspective.

        Context:
        {data['supporting_text']}

        Question:
        {pred_text}

        Guidelines:
        - Stay in character. Do not say "Based on the text" or "As the person described."
        - Use the first person ("I", "me", "my").
        - If the context doesn't have the answer, stay in character while saying you don't know.
        """
        
        response = llm_pipe.inference(prompt)
        response = response.replace('*', '')
        
        wav = tts_pipe.tts_infer(response)
        wav_path = os.path.join(output_path, f"sample_{index}.wav")
        sf.write(wav_path, wav, 24000)
        
        output_data.append(json.dumps({
            'wav_path': wav_path, 'gen_answer': response, 'question': data['question_text']
        }, ensure_ascii=False) + '\n')
        
        if index == 8:
            break
        
    with open(os.path.join(output_path, 'text.json'), 'w') as txtf:
        txtf.writelines(output_data)
    
if __name__ == "__main__":
    app()