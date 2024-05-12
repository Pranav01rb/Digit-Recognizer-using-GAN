# Design of a Real-Time GAN-based Speech Recognizer for Consumer Electronics

Authors: Pubali Roy, Pranav R Bidare, Priya Bharadwaj, Dr. Manikandan J

Paper: https://ieeexplore.ieee.org/document/10134295
# Overview
This paper presents the development of a real-time speech recognition system employing Generative Adversarial Networks (GANs) aimed at enhancing the functionality of consumer electronics. This work capitalizes on the rapid advancements in GAN technology, typically utilized for image processing, and adapts it for one-dimensional audio signals by converting them into two-dimensional spectrograms for processing.

# Methodology
The proposed system begins by capturing an audio signal through a microphone. These audio signals are initially processed to distinguish speech from ambient noise through an endpoint detection. It  uses energy thresholds to identify the active parts of the audio signal where speech is present. 

Once the speech segments are identified, they are converted into two-dimensional spectrograms. A spectrogram represents how the spectral density of a signal varies with time, an essential feature for speech recognition systems. This transformation is crucial as it translates audio signals into a format that can be effectively processed using two-dimensional GANs, which are typically used for image data.

The GAN model consists of a Generator and a Discriminator working in tandem to classify speech signals. The Generator crafts fake spectrogram images from random noise, while the Discriminator learns to differentiate between these fakes and real spectrograms, effectively classifying the speech.

The GAN is trained on a dataset comprising spectrograms of spoken digits. Each digit from 'zero' to 'nine' is uttered by multiple speakers to add diversity to the training data, enhancing the robustness of the model. The performance of the recognizer is evaluated based on its accuracy and speed in recognizing new speech inputs not seen during training.

The recognizer's output is tested under various conditions to ensure it performs well in real-world scenarios typical of consumer electronics environments, such as noisy backgrounds and different speaker accents.

<img width="400" alt="image" src="https://github.com/Pranav01rb/Digit-Recognizer-using_GAN/assets/57988947/a15a807b-3fbb-4ddf-97b0-77f966d15536">

Fig 1: Spectrogram of the input speech signal (Digit 0) 

<img width="400" alt="image" src="https://github.com/Pranav01rb/Digit-Recognizer-using_GAN/assets/57988947/2cb1e938-8ee2-47d5-8c95-b3baf377bb3f">

Fig 2: Spectrogram of the cropped speech signal (Digit 0)

# Results
The system demonstrated a remarkable maximum recognition accuracy of 100% with a processing time of 49.10ms per word. These results highlight the efficiency and potential applicability of GANs in real-time speech recognition tasks, specifically tailored for consumer electronics where prompt response times are crucial.
| Digit | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| 0     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 1     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 2     | 100.0%   | 0.88      | 1.00   | 0.94     |
| 3     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 4     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 5     | 100.0%   | 0.94      | 1.00   | 0.97     |
| 6     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 7     | 100.0%   | 1.00      | 1.00   | 1.00     |
| 8     | 93.33%   | 1.00      | 0.93   | 0.97     |
| 9     | 80.00%   | 0.92      | 0.80   | 0.86     |
| **Avg.** | **97.33%** | **0.974**  | **0.973**  | **0.974**    |

# How to Cite
@INPROCEEDINGS{10134295,
  author={Roy, Pubali and Bidare, Pranav and Bharadwaj, Priya and J, Manikandan},
  booktitle={2023 International Conference on Inventive Computation Technologies (ICICT)}, 
  title={Design of a Real-Time GAN based Speech Recognizer for Consumer Electronics}, 
  year={2023},
  volume={},
  number={},
  pages={1476-1480}

P. Roy, P. Bidare, P. Bharadwaj and M. J, "Design of a Real-Time GAN based Speech Recognizer for Consumer Electronics," 2023 International Conference on Inventive Computation Technologies (ICICT), Lalitpur, Nepal, 2023, pp. 1476-1480, doi: 10.1109/ICICT57646.2023.10134295.