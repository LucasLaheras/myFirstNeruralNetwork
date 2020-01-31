from Handler import Handler
import numpy as np
from CrossValidation import CrossValidation

if __name__ == '__main__':
    #Handler().write_data()

    X, y = Handler().read_datafile()
    y = np.asarray(y)
    y.reshape((y.size, 1))

    CrossValidation().validation(X, y, "Kfold")

