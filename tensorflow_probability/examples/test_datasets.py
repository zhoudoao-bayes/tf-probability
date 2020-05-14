# zhoudoao
# 2020.5.14


import sys 
sys.path.append('..')

import unittest

# from datasets. import mnist
from datasets.mnist import MNISTSequence
from datasets.cifar10 import CIFAR10Sequence


class TestMNIST(unittest.TestCase):
    def test_MNISTSequence(self):
        train_seq = MNISTSequence(batch_size=32, fake_data_size=6000)
        batch_x, batch_y = train_seq[0]
        self.assertEqual(batch_x.shape, (32, 28, 28, 1))
        self.assertEqual(batch_y.shape, (32, 10))
        
        heldout_seq = MNISTSequence(batch_size=64, fake_data_size=1000)
        batch_x, batch_y = heldout_seq[0]
        self.assertEqual(batch_x.shape, (64, 28, 28, 1))
        self.assertEqual(batch_y.shape, (64, 10))

        # self.assertEqual(train_seq, )
        # self.assertEquals(train_seq, heldout_seq, )


class TestCIFAR10(unittest.TestCase):
    pass


class TestCIFAR100(unittest.TestCase):
    pass




if __name__ == '__main__':
    unittest.main()
