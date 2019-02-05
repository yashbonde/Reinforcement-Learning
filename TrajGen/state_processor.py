'''
state_processor.py

@yashbonde - 02.02.2019
'''

class StateProcessor():
    def __init__(self):
        # Build the tensorflow graph for image processing
        with tf.variable_scope("Process_State"):
            self.input_state = tf.placeholder(tf.uint8, [210, 160, 3])
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
            
    def process(self, sess, state):
        '''
        Args:
            sess: tensorflow session
            state: a [210, 160, 3] RGB image
        Returns:
            a processed [84, 84, 1] greyscale image
        '''
        return sess.run(self.output, {self.input_state: state})