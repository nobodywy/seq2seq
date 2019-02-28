
import keras.backend as K
import  numpy  as np

a = np.asarray([1,2])
b = np.asarray([[3,4,5], [6,7,8]])
c = K.batch_dot(a,b,axes=1)
print()