# Walking Assessment Project.

## Project Description:

Among the balance tests, there are several walking tasks (TUG 4Meter-walk). They are difficult for the physiotherapist to determine time of walking.

In this project we determine automatically, the time of walking/number of frames of the walking tasks.

For the purpose of detecting the person in the video, We use MoveNet(Tensor Flow) Detection.


## Steps and Approaches:

### Detecting the correct person:
By pressing on the desired person, We can afterwards trace that person and analyze his walking.

### Detecting the ending line:
We take the coordinates sorrounding the floor (Decided by the foot coordinates of the detected person, and the camera), then We apply projective transformation
in order to get an image of the floor, which we apply Edge Detection and Hough transformation to detect lines, and we choose the line with the least common color.

Another approach is to determine it by letting the User press on the line, this is used in case the line can't be found automatically.

### Start and End time:
We try to find the starting time (The time the person starts walking), and the Ending time (The time the person crosses the line found previously), 
In order to analyze only the important frames.
