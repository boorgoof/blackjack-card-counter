# BlackJack Card Counter

**Authors:** Matteo Bino, Gianluca Caregnato, Federico Meneghetti  

## Project Overview

The project is a video analysis system capable of observing and interpreting the flow of cards dealt in a Blackjack game. Specifically, the system processes the video frames, overlays bounding boxes around detected playing cards, and performs a card count according to the Hi-Lo method. Each bounding box is color-coded based on the card's value in the following way:
- Green bounding boxes indicate cards valued at +1 (2 through 6).

- Blue bounding boxes indicate neutral cards with a value of 0 (7 through 9).

- Red bounding boxes indicate high-value cards that subtract from the count, assigned a value of -1 (10, face cards, and Aces).

## Run and Software Usage Mode

### Video Mode

The system receives as input a video of a Blackjack game and, after analysing 5 frames per second of the video, it returns a new video in which the detected bounding boxes are shown, and in the top right corner, the current game count is displayed according to the Hi-Lo method.

Furthermore, for each frame analysed, the system saves the corresponding image with the bounding boxes drawn around the detected playing cards and a `.txt` file containing the detected card labels in YOLO format. 

Note: no evaluation metrics are available in this usage mode.

To run this mode, simply execute the `run_video.sh` script:
```bash
    ./run_video.sh
 ```

### Video images (Model) Mode

The system analyzes 12 random video frames that were previously manually annotated. This allows evaluation of the model’s performance.

Therefore, in this usage mode, the system returns the images showing the detected bounding boxes, the files containing the annotations, and a metrics file reporting the Accuracy, the mean Average Precision (mAP), the mean IoU, and the precision, recall, and F1-score of the playing cards present in the images.

To run this mode, simply execute the `run_video_images.sh` script:
```bash
    ./run_video_images.sh
 ```
### Single Cards Model Mode

This mode is conceptually identical to the previous one (Video images Mode); however, the model analyzes the "Complete Playing Card Dataset" from Kaggle, which contains isolated card images without Blackjack context.

To run this mode, simply execute the `run_single_cards_model.sh` script:
```bash
    ./run_single_cards_model.sh
 ```
### Single Cards Mode

The system always analyzes the "Complete Playing Card Dataset" but using only traditional computer vision approaches (no deep learning model is used).

To run this mode, simply execute the `run_single_cards_template.sh` script:
```bash
    ./run_single_cards_template.sh
 ```


## Build

To build the project, run the `build.sh` script:

```bash
./build.sh
 ```
