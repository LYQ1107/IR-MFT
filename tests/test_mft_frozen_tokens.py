import unittest
import numpy as np
from tools.mft_frozen_tokens import square_crop,input_tensors


class TokenInputTests(unittest.TestCase):
    def test_bgr_to_rgb_and_neutral_padding(self):
        image=np.zeros((20,80,3),np.uint8);image[:]=[10,80,200]
        p,r=square_crop(image,[0,0,80,20],bgr=True)
        y,x,h,w=map(int,r);np.testing.assert_array_equal(p[y+h//2,x+w//2],[200,80,10])
        z,weights=input_tensors([p],[r],'cpu')
        self.assertTrue((z[0,:,:y,:]==0).all())
        self.assertAlmostEqual(float(weights.sum())*196,h*w,delta=.01)

    def test_scalar_and_repeated_gray_exact(self):
        image=np.arange(40*60,dtype=np.uint16).reshape(40,60).astype(np.uint8)
        a,r=square_crop(image,[3,4,55,35],bgr=False)
        b,s=square_crop(np.repeat(image[:,:,None],3,2),[3,4,55,35],bgr=True)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(r,s)

    def test_edge_box_and_fractional_patch_validity(self):
        image=np.full((31,71,3),127,np.uint8)
        a,r=square_crop(image,[-5,-5,25,26],bgr=True)
        z,w=input_tensors([a],[r],'cpu')
        self.assertEqual(tuple(z.shape),(1,3,224,224));self.assertEqual(tuple(w.shape),(1,256))
        self.assertTrue(((w>=0)&(w<=1)).all());self.assertTrue((w>0).any())


if __name__=='__main__':unittest.main()
