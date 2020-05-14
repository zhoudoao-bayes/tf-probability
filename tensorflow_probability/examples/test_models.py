# zhoudoao
# 2020.5

import sys
sys.path.append('..')

import unittest

from models.bayesian_lenet import bayesian_lenet


class TestModels(unittest.TestCase):
    def test_bayesian_lenet(self):
        image_shape = [28, 28, 1]
        model = bayesian_lenet(input_shape=image_shape)
        model.build(input_shape=[None, 28, 28, 1])
