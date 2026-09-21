# Assignment coverage

| Required step | Implemented solution | Evidence |
| --- | --- | --- |
| 1: Load MEL/NV images | Official ISIC labels, reproducible balanced sample, cached JPEG downloads | selected_manifest.csv |
| 1: Preprocess and segment | Blur, background color distance, Otsu, morphology, components; hair inpainting, bilateral filtering, CLAHE | lesions.py; segmentation examples |
| 1: Shape/color/texture features | PCA-aligned asymmetry, border irregularity, relative diameter, color statistics, LBP, masked GLCM | features.csv |
| 1: Classify and evaluate | Group-disjoint holdout, train-only grouped CV, SVM vs random forest | metrics.json, split_manifest.csv, test_predictions.csv |
| 2: Masked and full road images | Process all 8 provided images, road ROI and white/yellow paint | detections.json, annotated images |
| 2: Road lines and markings | Hough segments, intercept/slope clustering, vanishing-point filtering, paint gaps | lanes.py |
| 2: Lane regions and ego lane | Adjacent boundaries; centered-camera image-space ego point | annotated images; roadside views marked unavailable |
| 2: Cars and their lanes | Pretrained SSD with overlapping crops, NMS, box-bottom lane assignment | vehicle records and boxes |
| 2: Closeness and alerts | Relative bounding-box height; optional calibrated focal-length/assumed-height estimate in API | close_vehicle_alert; distance fields |
| 2: Line crossing goal | Temporal association, smoothing, hysteresis, persistence and cooldown | 90-frame video; synthetic crossing test |
| 3: Both RGB upper loops | 25 images per scene | scene contact sheets |
| 3: Features and matching | SIFT (or ORB), reciprocal ratio filtering, RANSAC fundamental matrix | pair_matches.json |
| 3: Closed-loop ordering | Global Hamiltonian-cycle heuristic with multiple starts and 2-opt | order.json and similarity plots |

Implemented extras: grouped leakage checks, safe archive recovery, video processing, small-object crops, alternative ORB descriptors and generic grayscale/Numpy multichannel loading. NIR, multispectral experiments, double-line recognition, video2 and joint lower/upper-loop reconstruction are not claimed as completed optional experiments.

The mandatory algorithms are implemented, but their predictions are imperfect. A still image cannot establish a past crossing event, a monocular uncalibrated image cannot establish metric distance, and a shared-feature graph alone does not certify the true physical image order.
