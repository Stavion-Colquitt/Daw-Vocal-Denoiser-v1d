> **Moved:** DAW Vocal Denoiser v1d now lives in **[Daw-Vocal-Denoiser/versions/v1d](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1d)**, together with every other version,
> one shared codebase and a results table across all versions. This repository is archived (read-only); its full history
> is part of Daw-Vocal-Denoiser.

# DAW Vocal Denoiser v1d

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

🤗 [Model on Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d) · v1: [Daw-Vocal-Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser) · v1c: [Daw-Vocal-Denoiser-v1c](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1c) · 🎧 [Live demo](https://huggingface.co/spaces/Stavion-Colquitt/Realtime_Denoisers) · 📋 [Project board](https://github.com/users/Stavion-Colquitt/projects/2)

Version 1d of my [DAW Vocal Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser): v1c's exact recipe (GRU 256, same data, same 108,779 steps) with one change, **5 ms of lookahead**: each frame's mask is computed after the model has heard one more 5 ms hop of input, for **15 ms algorithmic latency** instead of 10 ms. Same size as v1c (1.25M parameters). The question v1d asks: how much quality does 5 ms buy?

---

## Results (held-out test)

180 noisy clips (+ 20 clean pass-through clips) that no version trained on: my held-out song *Say Dat Name* (86 clips)
and held-out VocalSet singers (94 clips), mixed with dev-split noise at 0-40 dB SNR from a fixed seed. Every version
runs on exactly the same clips through the streaming engine (`compare.py` / `compare_multi.py`), so differences are paired.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | FAD-MERT (lower better) | Clean pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|
| Noisy input | 19.04 | 19.02 | 2.37 | 3.53 | 1.732 |  |
| v1 | 22.95 | 23.70 | 3.05 | 3.97 | **0.419** | 45.6 |
| v1c | 23.27 | 23.85 | 3.03 | 3.98 | 0.485 | **51.1** |
| v1b (GRU 384) | **23.46** | **24.07** | 3.10 | **3.99** | 0.448 | 47.6 |
| **v1d (5 ms lookahead)** | 23.43 | 24.03 | **3.11** | **3.99** | 0.446 | 50.5 |

**v1d vs v1c**, paired over the same clips (mean ± 95% CI; bold = the interval excludes zero):

| Clips | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL |
|---|---|---|---|---|
| All 180 clips | +0.17 ± 0.19 | +0.18 ± 0.18 | **+0.07 ± 0.02** | +0.01 ± 0.02 |
| My voice (86 clips) | +0.09 ± 0.24 | +0.14 ± 0.22 | **+0.08 ± 0.02** | **+0.01 ± 0.01** |
| Other singers, VocalSet (94 clips) | +0.23 ± 0.30 | +0.22 ± 0.28 | **+0.07 ± 0.03** | +0.01 ± 0.03 |

**What this shows:** 5 ms of lookahead buys about as much as doubling the model. v1d and [v1b](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser/tree/main/versions/v1b) (GRU 384, 2x the parameters) are tied within error on every metric, and both beat v1c on predicted quality (PESQ, all voices). v1d gets there at **half v1b's CPU cost and file size**, and keeps clean takes almost as untouched as v1c (50.5 vs 51.1 dB pass-through, v1b 47.6). The price is 5 ms more latency (15 ms total), still low enough to monitor through. Its SI-SDR gain over v1c (+0.17 dB) is not significant on 180 clips, so the clearest win is quality per CPU, not raw accuracy.

## Results (dev set)

| | v1 | v1c | v1b | v1d |
|---|---|---|---|---|
| GRU width | 256 | 256 | 384 | 256 |
| Algorithmic latency | 10 ms | 10 ms | 10 ms | 15 ms |
| Training steps | 108,773 | 108,779 | 108,779 | 108,779 |
| Dev SI-SDR gain, published checkpoint | +2.39 dB | +2.57 dB | +2.79 dB | +2.78 dB |

Published checkpoint: +2.78 dB (step 107,000, lowest dev loss; +2.79 dB at the last step). Training time: 9.3 h on an RTX 4060, Nov 19, 2026.

| Real-time property | Value |
|---|---|
| Algorithmic latency | 15 ms (10 ms windows + one 5 ms hop of lookahead) |
| CPU per 5 ms block (ONNX Runtime, desktop CPU) | 0.23 ms (real-time factor 0.046) |
| ONNX vs PyTorch parity (streaming engine) | 112.3 dB |
| Parameters | 1,250,563 |
| ONNX file | 5.0 MB |

---

## What changed from v1c

- **Model:** unchanged from v1c (GRU 256, 1.25M parameters).
- **Lookahead:** training pads the magnitude input by one frame and shifts the mask back one frame, so the mask for frame t comes from the GRU after it has seen frame t+1. The streaming engine (`stream.py`) mirrors this by holding each spectrum for one hop until its mask arrives. `test_lookahead.py` confirms the output depends on input exactly one hop ahead and no further; streaming vs. PyTorch parity is 112.3 dB.
- **Data and training:** identical to v1c (108,779 steps from scratch). Trained on an RTX 4060.
- **Code:** `DAW_LOOKAHEAD` defaults to 1 in `model.py` and `stream.py`, so the engine runs this model correctly with no settings.

---

## Files

| File | What it does |
|------|--------------|
| `model.py` | Model definition and asymmetric-window STFT (lookahead: `DAW_LOOKAHEAD`, default 1) |
| `stream.py` | Real-time block engine (240-sample blocks in and out) and `process_file` for offline use. numpy + onnxruntime only |
| `dataset.py` | On-the-fly noisy/clean mixing (SNR 0-40 dB, rooms, bleed, hiss, hum) |
| `prep_data.py`, `prep_own_extra.py` | Build the training set (same data as v1c) |
| `train.py` | Training loop (compressed spectral + multi-resolution STFT loss, AdamW, EMA 0.999, MLflow) |
| `run_v1d.ps1` | The training run, then ONNX export and parity check |
| `export_onnx.py` | Exports the one-frame streaming ONNX model |
| `check_parity.py` | Checks the streaming engine against PyTorch and measures CPU time per block |
| `test_lookahead.py` | Confirms the output depends on input exactly one hop ahead and no further |
| `eval.py` | Held-out test evaluation (all metrics below) |
| `compare.py`, `compare_multi.py` | Run several versions on identical clips (`compare_multi.py` handles each model's GRU width and lookahead) |
| `denoise.py`, `Dockerfile` | CPU-only container: CLI and small HTTP service |
| `results/` | Raw held-out comparison (`compare_test.md`, `compare_test.json` with per-clip scores) |

The weights (`daw_denoiser.onnx`, `daw_denoiser.pth`) are on [Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d) and attached to this repo's release. My own vocal recordings are not included.

## Usage

```python
# pip install onnxruntime numpy
from stream import StreamEngine          # expects daw_denoiser.onnx next to it
eng = StreamEngine()
y_block = eng.process(x_block)           # float32[240] at 48 kHz -> 240 denoised samples, 15 ms later

from stream import process_file          # offline: whole mono 48 kHz array in, denoised array out (time-aligned)
y = process_file(x)
```

## Evaluation metrics

`eval.py` runs the held-out test through the streaming engine and reports two groups of metrics, noisy input vs
enhanced output:

| Metric | What it measures | Why it's here |
|--------|------------------|---------------|
| SI-SDR (dB) | Waveform closeness to the clean take, ignoring level | Standard enhancement metric; the main number here |
| PESQ-WB | ITU-T P.862.2 predicted speech quality (at 16 kHz) | Kept for comparison with speech denoisers; it was designed for speech, so treat it as a rough guide on singing |
| Clean pass-through SI-SDR | How untouched clean vocals come out | The "do no harm" check for tracking |
| **SDR (BSS-Eval, 512-tap)** | Filter-based SDR, tolerant of small EQ/phase changes | Recommended over SI-SDR for music in recent singing-voice work |
| **ViSQOL v3, audio mode** | Full-band (48 kHz) perceptual quality, MOS-LQO 1 to 4.73 | The usual intrusive quality metric for music; Google's code, built from source in Docker (`tools/visqol`) |
| **MERT-MSE** | Distance between [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) music embeddings of output and clean take | Embedding metrics track listening tests better than SDR on singing |
| **FAD-MERT** | Fréchet distance between all outputs and all clean takes in MERT embedding space | Distribution-level check that outputs "sound like" clean vocals |

Before ViSQOL and MERT, every signal is loudness-normalized to -23 LUFS (EBU R128) so level differences don't bias
the scores. Results are also broken down (mean improvement with 95% confidence intervals) by input SNR band, noise
type (DNS noise, headphone bleed, hiss, hum) and vocal source, and every item is written to `eval_results_items.csv`.

```bash
pip install -r requirements-eval.txt          # torch is assumed to be installed already
docker build -t visqol:local tools/visqol      # one time; eval.py also builds it automatically if missing
python eval.py 200                             # held-out test
python eval.py 40 --split dev                  # check the pipeline on dev vocals without touching the test set
python eval.py 200 --no-visqol --no-mert       # speech-style metrics + SDR only (no Docker, no model download)
```

The test mixtures come from a fixed seed, so every run scores exactly the same clips. Objective metrics are still
only a proxy: a listening test (MUSHRA, ITU-R BS.1534) on real noisy takes is the final word.

References: Le Roux et al. 2019 (SI-SDR); Vincent et al. 2006 and Scheibler 2022 (BSS-Eval, fast_bss_eval);
Chinen et al. 2020 (ViSQOL v3); Li et al. 2023 (MERT); Kilgour et al. 2019 (FAD); Gupta et al. 2015,
"On the Perceptual Relevance of Objective Source Separation Measures for Singing"; "Teaching Speech Enhancement
Models To Sing" ([arXiv 2607.11630](https://arxiv.org/abs/2607.11630)) on SDR, MERT-MSE and loudness matching for singing.

## Docker

A CPU-only image (ONNX Runtime, no PyTorch) is on Docker Hub as [`stavion/daw-denoiser:v1d`](https://hub.docker.com/r/stavion/daw-denoiser/tags).
Any sample rate, mono or stereo; output is 24-bit WAV at the input's rate.

```bash
docker run --rm -v "${PWD}:/data" stavion/daw-denoiser:v1d vocal.wav vocal_clean.wav

docker run --rm -p 8000:8000 stavion/daw-denoiser:v1d serve
curl -X POST --data-binary @vocal.wav http://localhost:8000/denoise -o vocal_clean.wav
```

The container's output matches `stream.py` run directly (122.5 dB, limited by the 24-bit WAV output).

## Data paths

Every data location is an environment variable with a default inside the repo, so a fresh clone works without
editing code:

| Variable | Default | What goes there |
|----------|---------|-----------------|
| `DAW_DATA` | `data/daw-denoiser` | Training pool written by `prep_data.py` (own_vocals/, vocalset/, bleed/) |
| `DNS_DATA` | `data/dns` | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband data: `clean_fullband/` and `noise_fullband/` |
| `DAW_CORPUS` | `data/corpus` | Your own vocal takes, one folder per song (`<song>/*.wav`), read by `prep_data.py` |
| `DAW_MUSIC` | `data/ableton-music` | Ableton project folders scanned by `prep_own_extra.py` for extra vocal takes |

On Windows, set one for the current shell with `$env:DAW_DATA = "E:\my\data"` (PowerShell) or permanently with
`setx DAW_DATA "E:\my\data"`. On Linux/macOS: `export DAW_DATA=/my/data`. `data/` is gitignored.

## Training data and licenses

| Source | License |
|--------|---------|
| My own vocal takes and instrumentals | Mine (not distributed) |
| [VocalSet](https://zenodo.org/records/1442513) (Wilkins et al., ISMIR 2018) | CC BY 4.0 |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband speech and noise (subsets) | Per source, see the DNS repo |

No commercial sample packs. Code is MIT licensed (see `LICENSE`).

## Limits

- For monitoring and cleaning vocal takes (singing or speech) recorded in untreated rooms.
- Removes noise, not reverb. Mono, 48 kHz only (resample other rates first).
- Not tested on heavy distortion or SNR below 0 dB; very loud headphone bleed can survive.
- 15 ms of algorithmic latency: still low enough to monitor through, 5 ms more than v1, v1c and v1b.
