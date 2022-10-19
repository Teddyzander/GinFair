# GinFair
Uses a post-processing method to enforce group and individual fairness constraints on a pre-trained classifier.

## Post-processing for group Fairness
The optimisation process for enforcing group fairness on a pre-trained function is described by [Hardt et al (2016)](https://arxiv.org/pdf/1610.02413.pdf). It makes use of threshold optimisation and controlled randomness to enforce group fairness.

## Extension to Individual Fairness
The output space from the group fair optimisation is discontinuous at the threhsholds. Since we cannot treat 'similar individuals similarly' in these areas, individual fairness is violated.

<img src="https://user-images.githubusercontent.com/49641102/196636810-a49df217-4528-4ee7-8280-8c82ca608bb1.png" width="480">

We therefore opt to adapt the output space such that it is:
* Continuous
* Behaviour preserving
* Defined only by the pre-calculated parameters (thresholds and probabilities), avoiding any computation time

<img src="https://user-images.githubusercontent.com/49641102/196638036-50da6a67-3b2f-47a6-8250-348f190fb243.png" width="480">

We use a linear interpolant to creat a continuous space that improves individual fairness with minimal impact on group fairness. 

<img src="https://user-images.githubusercontent.com/49641102/196638369-c22e71bc-4557-480e-a59b-7c57c3d1634a.png" width="600">

## Instructions
The code is currently very bare bones, and is jsut a proof of concept.

To run Income Data analysis, set `synth = False`. For synthetic data, set `synth = True` (line 24)

To run for different fairness constraints, look to lines 27 and 28. Change the metric and constraint to any provided by [Fairlearn](https://fairlearn.org/main/api_reference/fairlearn.metrics.html)
