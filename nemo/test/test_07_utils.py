''' Unit tests for functions in nemo.utils.py '''


import unittest 

from nemo.utils import multiproc, multithread



def multiply(x, y = 2):
    return x * y


def multiply2(x, y, z = [3, 4, 5]):
    return z


class TestUtils(unittest.TestCase):

    def test_multiproc_ValueError_iterables_not_equal_len(self):
        '''
        Check designated iterables have equal length.
        '''

        def test_func(number1, number2):
            addition = number1 + number2

        with self.assertRaises(ValueError):
            multiproc(
                test_func,
                ['number1', 'number2'],
                n_workers = 10,
                number1 = list(range(100)),
                number2 = list(range(10))
            )     


    def test_multiproc_n_workers_geq_items(self):

        x = list(range(50))
        y = list(range(1, 51))
        result = multiproc(
            func = multiply,
            iterator_keys = ['x', 'y'],
            n_procs = 100,
            x = x,
            y = y
        )

        self.assertTrue(all([rv == xv * yv for xv, yv, rv in zip(x, y, result)]))


    def test_multiproc_n_workers_lt_items(self):

        x = list(range(100))
        y = list(range(1, 101))
        result = multiproc(
            func = multiply,
            iterator_keys = ['x', 'y'],
            n_procs = 10,
            x = x,
            y = y
        )

        self.assertTrue(all([rv == xv * yv for xv, yv, rv in zip(x, y, result)]))


    def test_multiproc_non_iterable_arg_list(self):
        x = list(range(100))
        y = list(range(1, 101))
        z = list(range(5))
        result = multiproc(
            func = multiply2,
            iterator_keys = ['x', 'y'],
            n_procs = 10,
            x = x,
            y = y,
            z = z
        )

        self.assertTrue(all([rv == zv for rv, zv in zip(result, [z for _ in range(100)])]))


    def test_multithread_ValueError_iterables_not_equal_len(self):
        '''
        Check designated iterables have equal length.
        '''

        def test_func(number1, number2):
            addition = number1 + number2

        with self.assertRaises(ValueError):
            multithread(
                test_func,
                ['number1', 'number2'],
                n_workers = 10,
                number1 = list(range(100)),
                number2 = list(range(10))
            )     


    def test_multithread_n_workers_geq_items(self):

        x = list(range(50))
        y = list(range(1, 51))
        result = multithread(
            func = multiply,
            iterator_keys = ['x', 'y'],
            n_procs = 100,
            x = x,
            y = y
        )

        self.assertTrue(all([rv == xv * yv for xv, yv, rv in zip(x, y, result)]))


    def test_multithread_n_workers_lt_items(self):

        x = list(range(100))
        y = list(range(1, 101))
        result = multithread(
            func = multiply,
            iterator_keys = ['x', 'y'],
            n_procs = 10,
            x = x,
            y = y
        )

        self.assertTrue(all([rv == xv * yv for xv, yv, rv in zip(x, y, result)]))


    def test_multithread_non_iterable_arg_list(self):
        x = list(range(100))
        y = list(range(1, 101))
        z = list(range(5))
        result = multithread(
            func = multiply2,
            iterator_keys = ['x', 'y'],
            n_procs = 10,
            x = x,
            y = y,
            z = z
        )

        self.assertTrue(all([rv == zv for rv, zv in zip(result, [z for _ in range(100)])]))



if __name__ == '__main__':
    unittest.main(verbosity = 2)