import numpy as np
from Handler import Handler
from Train import Train
import tensorflow as tf
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, cross_val_score
import cv2
import time


def mostra(img, name='Name'):
    img1 = img.copy()

    img1 = img1.astype(np.uint8)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class CrossValidation():
    def __init__(self):
        self.training_data = []

    def LPO(self):
        return LeavePOut(p=50)

    def LOO(self):
        return LeaveOneOut()

    def Kfold(self):
        return KFold(n_splits=5)

    def verifyType(self, str):
        if str == "Kfold":
            return self.Kfold
        elif str == "LOO":
            return self.LOO
        elif str == "LPO":
            return self.LPO


    def validation(self, X, y, typeValidation="Kfold"):

        typeValidation = self.verifyType(typeValidation)

        #X = tf.keras.utils.normalize(X, axis=1)  # os numeros das imagens ficam entre [0, 1]
        #X = X.reshape((250, Handler().img_size, Handler().img_size))
        """""

        X.reshape((250, Handler().img_size, Handler().img_size))
        #y.reshape((250))

        classif = KerasClassifier(build_fn=Train().train, epochs=50, batch_size=10)
        result = cross_val_score(estimator=classif, X=X, y=y, cv=5, scoring="accuracy")

        media = result.mean()
        print(media)
        desvio = result.std()
        print(desvio)

        """""

        #print(X, y)

        try:
            validation = typeValidation()
        except:
            print("does not exist")
            return

        suma = []
        sumb = []

        #mostra(X[1])

        start_time = time.time()

        for train_index, test_index in validation.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #for i in range(250):
            #    mostra(X_train[i])

            T = Train()
            T.trainU(X_train, y_train)
            a, b = T.evaluate(X_test, y_test)
            suma.append(a)
            sumb.append(b)

        print("--- %s minutos ---" % ((time.time() - start_time)/60))
        suma = np.asarray(suma)
        sumb = np.asarray(sumb)
        print("PERDA")
        print("MEDIA: " + str(suma.mean()) + " DEVIO PADRAO: " + str(suma.std()))
        print("CURACIDADE")
        print("MEDIA: " + str(sumb.mean()) + " DEVIO PADRAO: " + str(sumb.std()))

        return suma.mean(), sumb.mean()
