import pandas as pd
import numpy as np
# tfidf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model


class ContentBase:

    @staticmethod
    def getTfidf(x_train):
        transformer = TfidfTransformer(smooth_idf=True, norm='l2')
        tfidf = transformer.fit_transform(x_train.tolist()).toarray()
        return tfidf

    @staticmethod
    def get_items_rated_by_user(rate_matrix, user_id):
        if (rate_matrix.size == 0):
            return ([], [])
        y = rate_matrix[:, 0]
        ids = np.where(y == user_id)
        item_ids = rate_matrix[ids, 1];
        scores = rate_matrix[ids, 2]
        return item_ids, scores

    @staticmethod
    def getIndexInArr(index_arr, list_item):
        return list(map((lambda x: np.where(index_arr == x)[0][0]), list_item))

    @staticmethod  # Find the model for each user
    def GetRidgeRegression(self, n_users, rate_train, tfidf, W, b, index_arr):

        for n in n_users:
            i = np.where(n_users == n[0])[0][0]
            ids, scores = self.get_items_rated_by_user(rate_train, n[0])
            clf = Ridge(alpha=0.01, fit_intercept=True)
            
            if ids.size == 0:
                W[:, i] = 0
                b[0, i] = 0

                # return W, b
                continue

            tests = self.getIndexInArr(index_arr, ids[0])
            Xhat = tfidf[tests, :]
            if Xhat.size != 0:
                clf.fit(Xhat, scores[0])
                W[:, i] = clf.coef_
                b[0, i] = clf.intercept_
            else:
                W[:, i] = 0
                b[0, i] = 0

        return W, b

    # @staticmethod
    # def RMSE(self, n_users, rate_train, Yhat):
    #     se = 0
    #     cnt = 0
    #     print(n_users.shape[0])
    #     for n in range(n_users.shape[0] - 5):
    #         ids, scores_truth = self.get_items_rated_by_user(rate_train, n)
    #         scores_pred = Yhat[ids, n]
    #         e = scores_truth[0] - scores_pred[0]
            
    #         se += (e*e).sum(axis = 0)
    #         print("se", se)
    #         cnt += e.size 
    #     return np.sqrt(se/cnt)