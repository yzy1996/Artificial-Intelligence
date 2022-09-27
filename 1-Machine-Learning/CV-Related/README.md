computer vision tasks including:

- motion estimation
- 



Tracking objects (at the level of segmentation masks or bounding boxes)

Tracking certain points in certain categories (e.g., the joints of a person)

general-purpose fine-grained tracking.

the dominant approaches are feature matching (compute a feature for the target on the first frame, then compute features for pixels in other frames, and then compute “matches” using feature similarity (i.e., nearest neighbors).) and optical flow (compute a dense “motion field” that relates each pair of frames, and then do some post-processing to link the fields together.)







[Tracking Any Pixel in a Video](https://blog.ml.cmu.edu/2022/09/09/tracking-any-pixel-in-a-video/)

[Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories](https://arxiv.org/abs/2204.04153)

https://github.com/aharley/pips