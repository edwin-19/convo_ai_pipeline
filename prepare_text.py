import typer
from datasets import load_from_disk
import os
from utils import clean_text

app = typer.Typer()

@app.command()
def main(
    output_dir:str=typer.Option("data/"),
    dataset_name:str=typer.Option("data/context_dialog_subset_100"),
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataset = load_from_disk(dataset_name)
    texts = []
    
    for d in dataset:
        texts.append(d['question_text'])
        texts.append(d['answer_text'])
        texts.append(d['supporting_text'])
    
    texts_clean = [clean_text(text) + '\n' for text in texts]
    with open(os.path.join(output_dir, 'sample.txt'), 'w') as txtf:
        txtf.writelines(texts_clean)
    
if __name__ == "__main__":
    app()