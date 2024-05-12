# Design of a Real-Time GAN-based Speech Recognizer for Consumer Electronics

Authors: Pubali Roy, Pranav R Bidare, Priya Bharadwaj, Dr. Manikandan J
Paper: https://ieeexplore.ieee.org/document/10134295
# Overview
This paper presents the development of a real-time speech recognition system employing Generative Adversarial Networks (GANs) aimed at enhancing the functionality of consumer electronics. This work capitalizes on the rapid advancements in GAN technology, typically utilized for image processing, and adapts it for one-dimensional audio signals by converting them into two-dimensional spectrograms for processing.

# Methodology
The proposed system begins by capturing an audio signal through a microphone, which is then processed to distinguish between speech and ambient noise. Valid speech signals are transformed into a two-dimensional spectrogram that serves as the input for the GAN model. The GAN model consists of a Generator and a Discriminator working in tandem to classify speech signals. The Generator crafts fake spectrogram images from random noise, while the Discriminator learns to differentiate between these fakes and real spectrograms, effectively classifying the speech.

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