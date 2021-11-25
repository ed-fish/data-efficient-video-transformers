import unittest
import sys

sys.path.insert(1, '/home/ed/PhD/mmodal-moments-in-time')

from transforms.spatio_cut import SpatioCut
from transforms.img_transforms import ImgTransform



test_vid = "/home/ed/PhD/mmodal-moments-in-time/input/juggling/juggling.mp4"

class TestRandCrop(unittest.TestCase):
    def test_cutvid(self):

        sp = SpatioCut()
        output = sp.cut_vid(test_vid, 16)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(output[0]), 16)
        self.assertEqual(len(output[1]), 16)
        self.assertEqual(len(output[2]), 16)


if __name__ == '__main__':
    unittest.main()
