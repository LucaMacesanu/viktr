This is a subcomponent of the larger viktr project.
In this mini repo we want to reimlpement the data augmentation setup of the GR-RL paper to see if it can be used to boost model performance in our use case.
The augmentation pipeline is as follows:
During the offline training stage, we apply a simple yet effective morphological symmetry augmentation
paradigm, which further boosts the policy performance. The augmentation paradigm leverages the morphological symmetry in our bimanual task settings. For image observations ot, we flip all the images horizontally,
then swap the images from the left wrist with those from the right wrist. All transformations in proprioception
states st and actions at are converted via mirror symmetry in the world frame, and then transformed back to
local wrist frames. We also flip the spatial description in the language instructions accordingly, e.g., changing
“the hole on the left” to “the hole on the right”. Empirically, the symmetry data augmentation can effectively
enhance the performance of the policy.


As a first step for our project, we would like to train pi05 with and without the morphological symmetry augmentation.
You should try to keep the code easy to integrade into the viktr taining pipieline, which makes heavy use of some baseline code in nyu-finger-robot.
The configs and task set should be the same as the yor-pi05-ablation-quantiles.yaml, except use the left and right arm data instead of only left (8 tasks total)
