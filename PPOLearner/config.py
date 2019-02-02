'''
config.py

@yashbonde - 01.02.2019
'''

class Config():
	def __init__(self, args):
		self.save_path = './experiments'

		self.set_from_args(args)

	def set_from_args(self, args):
		for k, v in args.__dict__.items():
			setattr(self, k, v)