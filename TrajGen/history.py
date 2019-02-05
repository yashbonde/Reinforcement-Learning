'''
history.py

@yashbonde - 02.02.2019
'''

class History():
    def __init__(self, config):
        self.config = config
        self.history = np.zeros([config.buffer_size, 84, 84, 4], dtype = np.uint8)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        return self.history

    def reset(sef):
        self.history *= 0