# Computer Vision Projects

Three implemented Python projects for the University of Padova Computer Vision 2025–26 assignments. Includes reproducible experiments, saved results, explanatory notebooks, and algorithm tests.

| Project | Implementation | Included experiment |
| --- | --- | --- |
| 1 · Skin lesion classification | Classical segmentation, shape/color/texture features, grouped SVM/random-forest selection | 300 real ISIC 2019 images: 150 MEL + 150 NV |
| 2 · Road understanding | Hough boundaries, paint-run classification, lane regions, pretrained vehicle detector, temporal crossing cues | All 8 road images and first 90 frames of video1 |
| 3 · Image sequence estimation | SIFT, reciprocal matching, RANSAC geometry, closed-loop optimization | All 25 upper-loop RGB images in each scene |

These are coursework baselines with measured limitations, not guaranteed perfect solutions. Read [the reports](docs/RESULTS.md) before interpreting the outputs. The lesion experiment is a subset benchmark; the road detector has missed vehicles and false boundaries; sequence accuracy cannot be quantified without a verified reference order.

## Setup

Use Python 3.12. From this folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Open the three files under `notebooks/` in VS Code with the Jupyter extension and select this Python environment. Their saved-result sections work without downloading datasets. Each notebook also contains an optional pipeline rerun cell. Saved cells were executed with the included in-process IPython runner (`python scripts/execute_notebooks.py`); normal Jupyter/VS Code execution is also supported.

## Download data and detector weights

```bash
python scripts/download_data.py roads sequences weights
python scripts/download_data.py isic --per-class 150
python scripts/download_data.py video
```

The scene archives total approximately 1.14 GB even though only the required RGB frames are extracted. Images, source archives and pretrained detector weights are not bundled. Downloads require internet access and may take several minutes. Existing downloaded files are reused. If Kaggle changes public access, download the specified dataset manually, retain the generated official CSV files, and place the selected JPEGs in `data/isic/images/`, then rerun the command.

The challenging archive served during this run lacked a valid ZIP directory. The downloader recovered all 25 required upper-loop RGB members, verifying each decompressed size and CRC. It fails if the required frame count is incomplete.

To use every MEL/NV image rather than a balanced subset, pass `--per-class 0` (substantially more disk space and runtime). Results from that larger run are not included here.

## Run the projects

```bash
python -m cvprojects.lesions features data/isic/manifest.csv --out results/project1/features.csv
python -m cvprojects.lesions train results/project1/features.csv --out results/project1
python -m cvprojects.lesions predict path/to/image.jpg results/project1/classifier.joblib

python -m cvprojects.lanes data/roads --weights data/weights --out results/project2
python -m cvprojects.lanes data/video1.mp4 --video --max-frames 90 --weights data/weights --out results/project2/video1

python -m cvprojects.sequence data/sequences/easy/upper_loop/rgb --out results/project3/easy
python -m cvprojects.sequence data/sequences/challenging/upper_loop/rgb --out results/project3/challenging
```

Project 3 also supports `--method orb`. Supply a verified JSON filename list with `--truth reference.json` to evaluate cycle-edge recall and position accuracy up to rotation and reversal. Filenames are identifiers, not ground-truth ordering.

## Repository contents

- `cvprojects/`: complete implementation, importable or runnable as modules.
- `scripts/download_data.py`: reproducible dataset and weight setup.
- `notebooks/`: explanation, actual saved results, and optional reruns.
- `results/`: feature table, grouped split, fitted classifier, predictions, figures, detections, annotated video, pair scores and estimated sequences.
- `docs/RESULTS.md`: methods, observed results, limitations and interpretation.
- `docs/REQUIREMENTS.md`: assignment-to-implementation mapping.
- `docs/SOURCES.md`: data and model attribution.
- `tests/`: synthetic checks for the underlying algorithms and safe extraction.

Only load `classifier.joblib` from a trusted source; joblib is a Python serialization format. The saved model is an educational image-classification experiment, not a clinical diagnostic tool. The road alerts are image-space heuristics, not a driving safety system.

## Add to GitHub

Extract the ZIP, open `computer-vision-projects`, and upload its contents into your repository. Keep the directory structure. The raw `data/` folder is excluded by `.gitignore`. You can also open the folder in VS Code, initialize Git, commit, and publish to your chosen GitHub repository.

If using the GitHub website, drag the extracted folders/files into **Add file → Upload files**, then commit. Upload the extracted contents rather than only the ZIP so GitHub displays the code, notebooks and README.
