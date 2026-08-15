This is a research project. The goal here is to develop the VLA portion of the pipeline following the example of RICL. You can also find a draft of our paper with mock results and an explanation of the method in this folder as vktr.pdf.

For reference I have started setting up RICL under 3rd party. It is unclear to me whether we should just start with their codebase, or run from scratch. I am inclined towards starting from scratch using openpi as the source folder (which is similar to what they did).
Whatever codebase we make and test here will then be moved to HPC for large scale traaining. So for version control I would like to use uv, keep python scripts in scripts, and high levle running tools in shells (which should be .sh files).

The key different with our method is as follows:
1. We need to support different retrieval metrics: vision, value, and vision + value.
2. For the MVP, we will use Robometer for value estimatoin. Robometer should ship with lerobot version 0.6.
