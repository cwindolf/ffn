import numpy as np
import logging
from random import getrandbits, randrange


class FakePool:
    '''
    Store old generated volumes so discriminators aren't learning from
    current generators.

    After
    github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py
    Feel like it could be "mixier" with the batches but hey.
    '''

    def __init__(self, pool_size=64):
        logging.info(f'FakePool({pool_size})')
        self.size = pool_size
        self.pool = []

    def query(self, fakes):
        if self.size == 0:
            return fakes

        choices = []
        for fake in fakes:
            if len(self.pool) < self.size:
                self.pool.append(fake)
                choices.append(fake)
            elif getrandbits(1):
                ind = randrange(self.size)
                choice = self.pool[ind]
                self.pool[ind] = fake
                choices.append(choice)
            else:
                choices.append(fake)

        return np.array(choices)
