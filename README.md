# Target Detecting System

[matches]: output/matches_1.jpg 
[shapes]: output/shapes_1.jpg 
[display]: output/display_1.jpg 
[display2]: output/display_2.jpg 
[center]: output/center_1.jpg 
[board]: test_img_6/aligned_shot_1.jpg

### Scripts
- `alignByRefImg.py`: detecting and extract target board by its geometric shape.
- `alignBySquares.py`: detecting and extract target board by ORB (oriented BRIEF) features and interest points.
- `sound.py`: take photo and save it to file when the sound from microphone hit a certain threshold. The threshold was set based on multiple experiments with sounds of rifle and surrounding environment.
- `score.py`: take extracted target board image as an input then detect the shot location and calculat the final score.

### Detecting the target board by geometric shapes
![detecting shapes][shapes]

### Detecting and extracting the target board by SIFT points
![matching reference images][matches]

### Extacted target board
<!-- ![target board][board] -->
<img src="test_img_6/aligned_shot_1.jpg" alt="board" width="50%"/>

### Processing images
![detecting target][center]

### Location of the bullet hole and the final score
Example shot 1:
![score1][display]
Example shot 2:
![score2][display2]

## Testing Scripts
```bash
# detecting the board by geometric shapes
python alignBySquares.py input.jpg output.jpg

# example
python alignBySquares.py test_img_6/shot_1.JPG test_img_6/aligned_shot_1.JPG

# detecting the board by ORB features
python alignByRefImg.py output.jpg input.jpg reference.jpg

# example
python alignByRefImg.py test_img_6/aligned_shot_1.JPG test_img_6/shot_1.JPG test_img_6/aligned_shot_0.jpg

# get final score
python score.py input.jpg

# example
python score.py test_img_6/aligned_shot_1.jpg
```

<!-- ## Improvements
-  -->