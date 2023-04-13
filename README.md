# Computer Vision Music Maker

This project will be similar to [BeatBlocks](https://beatblocks.app/), where Lego blocks are used to control a soundboard and create music.

The aim of this project is to gain a better understanding of:

-   Fast Object Recognition
-   Multi-object detection
-   Customizable speed control
-   UI and UX

The intention is to investigate two methods for finding out the type and location of the objects that will ultimately make the different noises:
- Decompose the image into sub-images and perform classification on these
    - This has the advantage of being simple, repetitive and predictable
- Implement a YOLO algorithm or similar to find all items and their locations in a single look at the frame
    - This has the potential to be massively faster, orders of magnitude better for the music, but also to go horribly wrong.

These route alternatives are shown in the below flowchart:
![Code Flow Diagram](Flow.png?raw=true "Image of a code flow diagram showing alternative solutions being attempted") 

# Sub Images Pathway

[array_test.py](array_test.py) implements the decomposition algorithm on a test image. In doing it provides both a proof of concept, and a handy reference guide for where each part of the image ends up in the resulting array / for loops.

> Written with [StackEdit](https://stackedit.io/).
