# Project Notes — EcoSort

## Project title
Smart Waste Classification & Recycling Assistant

## Problem statement
People may be unsure how to separate common household waste into suitable disposal streams. EcoSort is an educational prototype that predicts the likely material/object in a photo and recommends one of three main sorting groups: Recyclable, Organic or Residual.

## Aim
To build a beginner-friendly computer-vision application that accepts a waste-item photograph, identifies its likely material/object category, and translates that prediction into understandable sorting guidance.

## Objectives
1. Accept a photo through upload or camera capture.
2. Preprocess the image for a pretrained classifier.
3. Predict one of ten model classes: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes or trash.
4. Map the predicted class into Recyclable, Organic or Residual using transparent application rules.
5. Apply a cautious ambiguity check so low or closely tied results are marked for review rather than forced into a sorting group.
6. Display the detected material, sorting group or review status, guidance and model-score chart.
7. Maintain a temporary session history and allow CSV export.

## AI and application logic
The application integrates the third-party pretrained model `prithivMLmods/Augmented-Waste-Classifier-SigLIP2`, based on SigLIP2. Hugging Face Transformers loads the image processor and model; PyTorch performs inference. The label mapping is read from the model configuration.

After the model predicts a material/object class, a confidence-and-separation check first reviews the top score and its margin over the second class. If the top score is below 30% or the gap is below 8 percentage points, the app returns “Mixed / unclear — needs review” and advises the user to upload one item at a time. This is a conservative prototype rule, not a validated accuracy threshold. Otherwise, a separate rule-based mapping assigns the class to one of three sorting groups:

- Recyclable: cardboard, glass, metal, paper, plastic, clothes and shoes. For textiles/footwear, reuse, donation or dedicated collection is recommended; this does not guarantee curbside recycling acceptance.
- Organic: biological waste.
- Residual: general/mixed trash. Battery predictions appear in this branch only for the three-group overview and trigger a special warning to use an authorised battery collection point, never an ordinary residual bin.

This project integrates the pretrained classifier; it does not claim to have trained the underlying model from scratch.

## Syllabus mapping
- Unit II — digital image input and image preprocessing
- Unit III — image classification and Vision Transformer concepts
- Unit VI — integrated real-world AI application and evaluation metrics
- Practicals — image classification using a deep-learning framework

## Evaluation plan
Build a separate labelled collection of photos for each supported model category. Include varied lighting, backgrounds and camera angles, especially photographs taken on a phone. Record actual labels and predictions, then calculate accuracy, precision, recall and F1-score and inspect a confusion matrix. Do not claim a performance percentage until this evaluation has been conducted. Mixed-waste images should be tested separately because the model predicts one overall class per image.

## Limitations
- One predicted class is produced for the whole image; the application does not locate or separately sort multiple items in one mixed-waste photo. Ambiguous score patterns are flagged for review, but this does not turn the model into an object detector.
- A model score is not a calibrated probability of correctness. Confident wrong predictions are possible.
- Recycling acceptance differs by local authority and item condition.
- Clothes and shoes are mapped to a reuse/textile-recovery route under the Recyclable group; local services may not accept them in ordinary recycling.
- Batteries need special collection and must not be placed in ordinary residual waste.
- The application is an educational prototype, not a certified waste-management system.


## Visual design update
- Design direction: playful 3D illustration with a polished, presentation-ready interface.
- Color system: warm ivory, sage, mint, forest green, and soft color accents for waste materials.
- UI updates: redesigned hero area with a bundled 3D waste-bin illustration, raised workflow cards, softer upload and result panels, rounded metrics, and material-specific chart colors.
- Functional behavior retained: the model ID, image processor, prediction code, class mapping, confidence/review thresholds, three sorting groups, battery special-handling warning, session history, and CSV export remain unchanged.
