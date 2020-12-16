"""
This is a coarse version of Audio2Video implementation.
When training, image, which is used to provide appearance, is fed into the VAE-like image branch, and audio is fed into
an valina encoder. The combined code is the overall representation of the semantic, which will be decoded by a decoder
to generate corresponding video. In the test phase, the image branch is vectors from a Gaussian distribution.

The synthesized video contains the appearance of the input image, and its dynamic
property is defined by the audio. For instance, if the given audio is bird's singing, and the image is a description of
a bird in cage, then the generated video should be that bird is singing in a cage; however, if the given image is a bird
on a tree in a park, then the corresponding video should change accordingly.

And after this work, something related but maybe more interesting should also be tested. For example, generating video
through a piece of music. The model should be able to capture the content, i.e., what story is the piece sing about?
"""


