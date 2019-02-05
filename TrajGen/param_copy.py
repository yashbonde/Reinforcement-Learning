'''
param_copy.py

@yashbonde - 02.02.2019
'''

class ModelParametersCopy():
    def __init__(self, estimator1, estimator2):
        '''
        As per the paper we need to update the model after every few training iterations.
        Args:
            estimator1: estimator (Network) to copy values from
            estimator2: estimator (Network) to copy values to
        '''
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key = lambda v:v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key = lambda v:v.name)
        
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        
    def make(self, sess):
        return sess.run(update_ops)