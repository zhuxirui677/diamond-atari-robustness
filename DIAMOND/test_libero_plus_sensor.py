import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from diamond_libero_plus_runner import plus_conditions, transform_rgb, plus_sensor, FAMILIES

class PlusSensorTests(unittest.TestCase):
    def test_original_sigma_and_rgb_channels(self):
        image=np.zeros((64,64,3),dtype=np.uint8)
        image[32,32,0]=255
        for level in (1,5,10):
            actual=transform_rgb(image,'plus_gaussian_blur',level)
            expected=(gaussian_filter(image.astype(float)/255,sigma=(level,level,0),mode='nearest')*255).astype(np.uint8)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual[:,:,1:].max(),0)
    def test_sham_bypasses_quantization(self):
        sentinel=object()
        self.assertIs(plus_sensor(sentinel,'sham',0,None),sentinel)
        self.assertIs(plus_sensor(sentinel,'normal',0,None),sentinel)
    def test_rejects_non_upstream_levels(self):
        image=np.zeros((64,64,3),dtype=np.uint8)
        for value in (0,11,1.5):
            with self.assertRaises(ValueError): transform_rgb(image,'plus_gaussian_blur',value)
    def test_full_includes_original_ten_levels(self):
        self.assertEqual([s for _,s in plus_conditions()[2:]],list(range(1,11)))
        self.assertEqual(len(plus_conditions(True)),4)
    def test_all_family_condition_sets(self):
        for family in FAMILIES:
            self.assertEqual(len(plus_conditions(False,family)),12)
            self.assertEqual(plus_conditions(True,family)[-1],('plus_'+family,10.))
    def test_stochastic_functions_replay_without_global_rng_leak(self):
        image=np.random.default_rng(4).integers(0,256,(64,64,3),dtype=np.uint8)
        for family in ('fog','glass_blur','zoom_blur'):
            np.random.seed(37)
            expected=np.random.random(3)
            np.random.seed(37)
            a=transform_rgb(image,'plus_'+family,1,np.random.default_rng(9))
            np.testing.assert_array_equal(np.random.random(3),expected)
            b=transform_rgb(image,'plus_'+family,1,np.random.default_rng(9))
            np.testing.assert_array_equal(a,b)
            self.assertFalse(np.array_equal(a,image))

if __name__=='__main__': unittest.main()
