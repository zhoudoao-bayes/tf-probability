# zhoudoao
# 2020.5

import sys
sys.path.append('..')

import unittest

from models.bayesian_lenet import bayesian_lenet


class TestModels(unittest.TestCase):
    def test_bayesian_lenet(self):
        model = bayesian_lenet()
        model.build(input_shape=[None, 28, 28, 1])
        
        self.assertEqual(model.output, )
        


