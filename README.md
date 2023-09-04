# Walking Assessment Project.

## Project Description:

Among the balance tests, there are several walking tasks (TUG 4Meter-walk). They are difficult for the physiotherapist to determine time of walking.

In this project we determine automatically, the time of walking/number of frames of the walking tasks.

For the purpose of detecting the person in the video, We use MoveNet(Tensor Flow) Detection.


## Steps and Approaches:

### User-interface:
We have designed a user-friendly interface that allows users to run the application and select their desired video without needing to make changes within the code.

![result2](https://github.com/Abedulftah/walkingAssessment/assets/82156892/266f701d-f31a-4133-9154-ea60ed5e8433)


### Detecting the target person:
By pressing on the target person, We can afterwards trace that person and analyze his walking.

### Detecting the start and finish lines:
We take the coordinates sorrounding the floor (Decided by the foot coordinates of the detected person, and the camera), then We apply projective transformation
in order to get an image of the floor, which we apply Edge Detection and Hough transformation to detect lines, and we choose the furthest and third furthest lines as the start and finish lines respectivly, since usually there will be another line between them, the distance between the start line and finish line is 4 meters.

Another approach is to determine it by letting the User press on the lines, this is used in case the lines can't be found automatically, in case only one line is detected the user only has to choose the start line.

### Start and End time:
We try to find the starting time (The time the person starts walking), and the Ending time (The time the person crosses the line found previously), 
In order to analyze only the important frames.

### Tracking the person's movement:
While moving, between each consecutive frames the program keeps tracking the person with the least error for the coordinates of the frames (We give more weight to the vertical distance, since We assume the person is moving forward, so vertically the error shouldn't change much).
