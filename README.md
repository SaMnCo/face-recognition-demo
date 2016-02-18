# Face Recognition on the edges
## Intro

Face Recognition is not a new technology. It's been around for a long time. However, when it happens at scale, using millions of cameras spread across a city or a country, bandwidth requirements can be pretty intensive. At some point, there is a limit in scale, as the cloud would not be able to sustain the rate of images and the number of feeds. 

To solve this problem, a solution is to move compute where it can first happen: on the edges. Providing each camera with a locally adapted model of what it should or should not see allows to offload the cloud from 90% of computation tasks, and reduces the requirements on the bandwidth. 

To make this possible, a new paradigm of applications is necessary, that combine a training part in the cloud with the ability to run pre-trained models locally on cameras. 

Our application is a demonstration of such a workflow. 

## Training in the cloud

The first part of the model is an app that takes a number of pictures of people we want to recognize, under different conditions: happiness, sadness, winking, wearing glasses... The more pictures are available the better. 

The output of that process is a XML file that contains vectors of the neural network, based on HAAS cascades. 

## Shipping the model to the edges



