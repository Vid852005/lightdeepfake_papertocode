# LightFakeDetect: Optimised CPU Implementation 

## Abstract
This project is an optimised implementation of the **LightFakeDetect** architecture, originally proposed by AlMuhaideb et al. (2025). The goal was to reproduce a state-of-the-art deepfake detection pipeline (MTCNN + MobileNet + CBAM + GRU) and re-engineer it to run efficiently on consumer-grade hardware (**Ryzen 5 7520U, 16GB RAM**). 
Th dataset used for both the original and my pipeline is [Celeb-df-v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)

## Adaptations made
To adapt the paper's GPU-centric approach for a CPU environment, I implemented these structural and pipeline changes:

*  froze the majority of the MobileNet backbone and selectively unfreezing only the terminal blocks, I achieved a massive speedup without losing feature extraction quality.
*  Developed a custom data-loading pipeline to handle 1000+ videos. This prevents OOM (Out of Memory) crashes by streaming processed frames in balanced batches.
*  Experimented with spatial trade-offs, determining that **160x160** provides the optimal balance between inference speed (1.45 FPS) and detection accuracy.

## Methodology
The pipeline consists of four primary stages:
1.  **Face Extraction:** MTCNN isolates facial regions to remove background noise.
2.  **Feature Mapping:** MobileNetV2 extracts spatial features from the face crops.
3.  **Attention (CBAM):** A Convolutional Block Attention Module highlights subtle cues in the eyes and mouth.
4.  **Temporal Analysis:** A GRU processes sequences of 40 frames to detect inter-frame relation which is relatively less in deep fake or does not exist at all.

## Experiments & Results
Evaluation was performed on a [balanced dataset of real and manipulated videos](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2).


* **Accuracy** :( **74% –> 78%** )Improvement was observed as the dataset scaled from 500 to 1000 videos. 
* **Inference Speed**( **1.45 FPS**) **5 times faster** than original on CPU. 
* **Temporal Context**:(**40 Frames**)Increased from 20 to 40 for better anomaly detection. 
* **Precision** : (**~80%** ) reduced false positives because of balanced sampling.

## Challenges
* I had to work out how many changes I can make before I have gone too far to call this implementation of this research.
* I was achieving high accuracy but high hallucination and because I could see only accuracy and was not printing out the confusion matrix I though the results were good.
* Unbalanced sample size was the major reason for hallucination.

## References
[AlMuhaideb, S., Alshaya, H., Almutairi, L., Alomran, D., & Alhamed, S. T. (2025). LightFakeDetect: A Lightweight Model for Deepfake Detection in Videos That Focuses on Facial Regions. Mathematics, 13(19), 3088.](https://doi.org/10.3390/math13193088)
