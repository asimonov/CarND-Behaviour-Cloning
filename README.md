# Project 3: behavioural cloning
Alexey Simonov's submission for the Udacity Self-Driving Car Project 3: Behavioral Cloning.

## Overview
model.py was created to train convolutional neural network to
drive first track in
Udacity's Self Driving Car (SDC) Simulator.

## Approach to solving the problem

The approach I took was to use Udacity's provided driving data as I had no joystick.
As many people on the forum have found collecting the data using keyboard inputs
does not produce good training results.
But using the provided data alone does not yield good results.

So I took inspiration from [Vivek Yadav post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.52dqz7k51)
and generated extra augmented images to train the model for 'recovery' situations
where the model sees images shifted up/down and left/right randomly, with correspondingly
adjusted steering angles.
I feed original center images from udacity set, along with same images flipped horizontally.
These two sets of images are shuffled and original and flipped versions do not go to same
batches, otherwise model just trains to stay straight.
The recovery set of images is same lenght as originals, so effectively for ~8000 original
images I feed 3 times or about 24000 images to train the model.

I have used resized images, halving both width and height to make the model faster.

I have also masked top and bottom of each image so the model learns from what is truly
relevant for driving -- the view of the road and not the bonnet or the sky features.

I have experimented with the model inspired by [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
and with the model provided by [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py).
Of the two I have chosen NVIDIA as it was simpler and gave better results.

## Model Architecture

The final network architecture has about 890k parameters and the following layers:

1. normalization (Lambda x/127.5-1.0)
2. convolution 5x5 with 24 output channels and 2x2 stride, relu activations
3. convolution 5x5 with 36 output channels and 2x2 stride, relu activations
4. convolution 5x5 with 48 output channels and 2x2 stride, relu activations
5. convolution 3x3 with 64 output channels, relu activations
6. convolution 3x3 with 64 output channels, dropout 10%, relu activations
7. fully connected layer with 300 units, rely activation
8. fully connected layer with 10 units, no activation
9. output layer with 1 unit, no activation


## Training

If trained with 5 epochs of ~24k images as described above this model produces
reasonable track driving experience, but there are two places where it touches the kerb
-- first steep corner to the left after the bridge and subsequent steep right corner.
It still manages to drive them in most random initializations.
I am submitting the model that was trained this way in model.json and model.h5.

I have then used 'live trainer' shared by [Thomas Anthony](https://github.com/thomasantony/sdc-live-trainer)
to fine-tune my model on the two problematic corners.
This tool is very useful as I was just driving about 5 laps, training the model
incrementally only on slow drive data around those two corners.
It improved the model.
The 'live trainer' also allows for generating/gathering smoother steering data
as it decays steering angle from the keyboard in small increments.

I have included the 'live trainer' with my submission. The fine-tuned model there is in
checkpoin.h5 file -- you need to rename it to model.h5 to see how it runs on track.

### model.py

The file that defines the model, loads the gata, creates generators that produce
augmented data, trains the model and saves resulting weights into model.h5

### model-nvidia-gen.ipynb

Is the notebook version of model.py with image visualisations

### drive.py

The modified drive.py provided by Udacity.
I resize the images and apply the mask in it.
I also control the throttle in a very rough way to slowdown for bigger steering angles.


# Usage

Run 'python drive.py model.json' to control the simulator.
For the usage of 'live trainer' see Thomas Anthony page. My version is only modified to 
preprocess images so they are understood by my model.



