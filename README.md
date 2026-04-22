# Conversational Audio Pipeline

This project runs a small conversational audio workflow:

1. Download a subset of the ContextDialog dataset
2. Generate a reference voice sample for TTS
3. Prepare cleaned text data - this is for 
4. Run end-to-end inference to produce generated answers and audio files

## Prerequisites
- install KenLM [Install](https://github.com/kpu/kenlm/blob/master/lm/builder/README.md)
- install espeak [Install](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)

## Install Dependencies

Create and activate a virtual environment, then install packages:

```bash
pip install -r requirements.txt
```

If your environment already has CUDA/PyTorch pinned, keep your existing Torch install and only install the remaining packages.

## Quick Start

### 1) Download a dataset subset

```bash
python download_subset.py download --save-path data/context_dialog_subset_100 --num-samples 100
```

### 2) Generate TTS reference files

```bash
python download_subset.py gen-ref --dataset-name data/context_dialog_subset_100 --ref-path ref
```

This creates:

- `ref/ref.wav`
- `ref/ref.txt`

### 3) (Optional) Export cleaned text

```bash
python prepare_text.py --output-dir data --dataset-name data/context_dialog_subset_100
```

This creates `data/sample.txt`.

### 4) Run inference

```bash
python infer.py \
  --dataset-name data/context_dialog_subset_100 \
  --ref-path ref \
  --output-path output \
  --device cpu
```

By default, the script processes up to 9 samples and writes:

- `output/sample_<index>.wav`
- `output/text.json`

## Notes

- Keep your dataset path consistent between commands.
- Ensure `ref/ref.wav` and `ref/ref.txt` exist before running `infer.py`.
- If audio write errors occur, verify that `soundfile` and system audio dependencies are installed.
- Most of the models were downloaded locally 

## TODo
- [ ] Streaming Inference