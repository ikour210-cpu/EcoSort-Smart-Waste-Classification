# EcoSort — Smart Waste Classification & Recycling Assistant

EcoSort is an initial-stage computer-vision web application. Upload or capture a photo of **one waste item**. A pretrained image classifier predicts the likely material/object class. EcoSort then maps that class into one of three main sorting groups:

1. **Recyclable** — recyclable materials and reuse/textile-recovery suggestions where relevant.
2. **Organic** — biodegradable food or plant material suitable for composting/organic collection where available.
3. **Residual** — material that is not suitable for ordinary recycling or composting.

Battery predictions appear under the Residual branch for the three-way overview, but the app displays a prominent special-handling warning: **batteries must not go into ordinary residual waste** and should be taken to an authorised battery collection point.

## Supported model classes

- Battery
- Biological / organic material
- Cardboard
- Clothes / textiles
- Glass
- Metal
- Paper
- Plastic
- Shoes
- General / mixed trash

## Files

- `app.py` — Streamlit application, model inference, and three-way sorting logic
- `assets/ecosort_hero_3d.png` — bundled 3D illustration used in the interface
- `requirements.txt` — Python dependencies
- `setup_and_run.bat` — creates a virtual environment, installs dependencies and starts the app
- `run_app.bat` — starts the app after setup
- `PROJECT_NOTES.md` — project explanation, syllabus mapping and evaluation plan

## Run on Windows 10/11

1. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ . Select **Add Python to PATH** during installation.
2. Extract the project folder.
3. Double-click `setup_and_run.bat` once. It creates `.venv`, installs dependencies and launches Streamlit.
4. On later runs, double-click `run_app.bat`.
5. On the first classification, the app downloads the pretrained model from Hugging Face. Keep an internet connection active. Model files are cached locally for later runs.
6. Upload or capture one clear photo of one item and select **Classify and sort**.

## How it works

1. Streamlit accepts an uploaded or camera-captured image.
2. Pillow opens the image and converts it to RGB.
3. The Hugging Face image processor prepares the image for the pretrained classifier.
4. The SigLIP2-based classifier runs inference through PyTorch and returns class scores.
5. EcoSort reads the predicted material/object label from the model configuration.
6. A confidence-and-separation check reviews the top model score and the gap to the second score. Low or closely tied results are marked **Mixed / unclear — needs review** instead of being forced into a bin.
7. For sufficiently clear predictions, a transparent rule-based mapping assigns the label to Recyclable, Organic or Residual.
8. The application displays the material, sorting group (or review status), disposal guidance and model-score chart.
9. A temporary session history is available and can be exported as CSV.

## Pretrained model and attribution

This version integrates [`prithivMLmods/Augmented-Waste-Classifier-SigLIP2`](https://huggingface.co/prithivMLmods/Augmented-Waste-Classifier-SigLIP2), a third-party pretrained image classifier based on SigLIP2. Its model card lists ten classes: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes and trash. The app downloads the model on first use. Review the model card and licence before reuse.

**Important project distinction:** the pretrained classifier supplies the material/object prediction. The three-way sorting groups and the guidance mapping are application logic written in this project. The underlying pretrained model has not been trained from scratch by this project.

## Important limitations

- The model predicts one label for the entire image. It does not detect, count or separately classify multiple objects in mixed-waste photos. Photograph one item at a time for the intended workflow. If the image is a mixed pile or the model's leading classes are too close, the app displays a review message instead of recommending Recyclable, Organic or Residual.
- The displayed score is a softmax model output, not a calibrated probability that the prediction is correct. A high score can still be wrong.
- As a cautious prototype rule, the app marks a result for review when the top score is below 30% or the gap between the top two scores is below 8 percentage points. These thresholds are initial guard rails, not scientifically validated cut-offs; review them using a labelled test set.
- A predicted material does not guarantee that an item is accepted by local recycling services. Follow local municipal and recycler rules.
- Clothing and shoes are mapped to the Recyclable group as a reuse/textile-recovery route; reuse or donation is preferred where possible. Ordinary curbside recycling may not accept them.
- Batteries are shown under the Residual branch only to keep the three-way overview. They require special collection and must never be put in ordinary residual waste.
- The app is an educational prototype, not a certified waste-management system. Its real-world performance should be evaluated with a separate labelled photo set before making accuracy claims.

## Suggested evaluation before submission

Prepare labelled photos from each model class, including your own phone photographs with varied backgrounds, angles and lighting. Record the actual class and prediction. Calculate accuracy, precision, recall and F1-score, and inspect a confusion matrix. Report only the results you measured yourself. Keep single-item photos separate from mixed-waste photos because the model predicts one label per image.

## Troubleshooting

**First run is slow:** the model is downloading from Hugging Face. Keep the internet connection active.

**Model download fails:** check internet access and restart the app.

**`streamlit` is not recognised:** activate `.venv` and run `python -m pip install -r requirements.txt`, then `python -m streamlit run app.py`.

**Camera is unavailable:** use the upload option. Camera support depends on browser permission and device support.


## Interface design

The interface uses a playful, polished 3D-illustration direction: a warm ivory and soft-sage canvas, forest-green navigation, rounded cards with gentle depth, color-coded material charts, and a locally bundled 3D waste-sorting illustration. The illustration is a static visual asset; it does not affect model inference or sorting decisions.
