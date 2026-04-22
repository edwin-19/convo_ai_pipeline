import typer
from datasets import load_dataset, Dataset, load_from_disk, Audio
import os
import soundfile as sf

app = typer.Typer()

@app.command()
def download(
    dataset_name:str=typer.Option("ContextDialog/ContextDialog"),
    save_path:str=typer.Option("context_dialog_subset_100"),
    num_samples:int=typer.Option(100)
):
    ds_stream = load_dataset(dataset_name, split='test', streaming=True)
    subset_list = list(ds_stream.take(num_samples))
    
    local_ds = Dataset.from_list(subset_list)
    print(local_ds)
    
    local_ds.save_to_disk(save_path)
    print(f"Success! 100 samples saved to: {os.path.abspath(save_path)}")
    
@app.command()
def gen_ref(
    dataset_name:str=typer.Option("./data/context_dialog_subset_100"),
    ref_path:str=typer.Option("ref")
):
    dataset = load_from_disk(dataset_name)
    dataset = dataset.cast_column("question_audio", Audio(sampling_rate=16000))
    dataset = dataset.cast_column("answer_audio", Audio(sampling_rate=16000))
    
    if not os.path.exists(ref_path):
        os.makedirs(ref_path)
    
    # 3. Pull the first sample
    data = dataset[0]
    
    print(f"Dataset loaded: {dataset}")
    print(f"Features: {dataset.features.keys()}")
    
    # sf.write('question.wav', data["question_audio"]["array"], samplerate=data["question_audio"]["sampling_rate"])
    
    sf.write(os.path.join(ref_path, 'ref.wav'), data["answer_audio"]["array"], samplerate=data["question_audio"]["sampling_rate"])
    with open(os.path.join(ref_path, 'ref.txt'), 'w') as txtf:
        txtf.writelines(data['answer_text'])
    
if __name__ == "__main__":
    app()