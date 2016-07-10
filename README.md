Overview
============
This is a  Learning Handwritten Character Classifer written in Python using the SciPy library. This is the code for the One-Shot Learning episode of Fresh Machine Learning on [Youtube](https://youtu.be/FIjy3lV_KJU). The code trains against a few examples of handwritten characters and then tries to classify characters correctly. The error rate is around 38%. If you want to try a state-of-the-art, better-than-human, one-shot learning library that you can apply to all sorts of data, check out [this](https://github.com/MaxwellRebo/PyBPL) repo. 

This is the Generative Adversarial Network Demo written in Python using the Theano ML library. This is the code for the Generative Adversarial Network Episode of Fresh Machine Learning on [Youtube](https://youtu.be/deyOX6Mt_As). There is a default static gaussian distribution curve that the generator curve continously tries to mimic. Each timestep it gets better and better as it slowly gets better at fooling the disciminator network. Here's a cool live [demo](http://cs.stanford.edu/people/karpathy/gan/)

Dependencies
============

* Python - (https://www.python.org/downloads/)
* matplotlib - `pip install matplotlib`
* theano - `pip install theano`
* scipy - `pip install scipy`
* foxhound (included)

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========
Step 1 - If your dependencies are installed you can just run the code! A pyplot GUI should pop up and it will get better and better as it trains. 

```shell
python demo.py
```

*Pull Requests are encouraged!!! This pyplot stuff can get wonky, so definitely make a PR if you see something that needs fixing.*

Credits
===========
Credit for this demo code Alec Radford and can be found [here](https://gist.github.com/Newmu/4ee0a712454480df5ee3). I've merely moved some functions around and added a lot more documentation. 
