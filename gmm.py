import numpy as np
from scipy.stats import multivariate_normal
import sys
sys.path.append("/Users/masumi/Desktop/Study/Sem5/SMAI/Assignments/A5/smai-m24-assignments-MasumiD/models/k_means")
from k_means import Kmeans
# from MinMaxScaling import MinMax
import numpy as np

class GMM:

    def __init__(self, c=2, max_iterations=1000):
        self.c = c
        self.max_iterations = max_iterations
        self.weights = None
        self.means = None
        self.covar = None
        self.r = None
        self.log_likelihood = None


    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0) + 1e-8
        X = (X - self.mean)/self.std # standardize
        # X = X.to_numpy()
        n, num_features = X.shape
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        ## initialising randomly

        # weights -> c random numbers
        # generate c random numbers and divide all of them by their sum so that it sums to 1
        self.weights = np.random.rand(self.c)
        denom = np.sum(self.weights)
        self.weights = self.weights/denom
        # print(f'weights: {self.weights}')
        min_values = np.min(X, axis=0)  
        max_values = np.max(X, axis=0)
        self.means = np.array([min_values + (max_values - min_values) * np.random.rand(num_features) for _ in range(self.c)])
        # print(f'means: {self.means}')
        # generate a random matrix and take A.(A.T) as covar matrix
        self.covar = []
        for _ in range(self.c):
            A = np.random.randn(num_features, num_features)  # Using standard normal distribution for random values
            self.covar.append(np.dot(A, A.T))  # it ensures a positive semi-definite covariance matrix
        self.covar = np.array(self.covar)


        ## initiaising using k means
        # self.covar = []
        # ini = Kmeans(k=self.c, max_iterations = 100)
        # self.means = ini.fit(X)
        # labels = ini.predict(X)
        # self.covar = np.zeros((n, num_features, num_features))
        # for k in range(self.c):
        #     cluster_points = X[labels == k]
        #     if len(cluster_points) > 0:
        #         self.covar[k] = np.cov(cluster_points, rowvar=False) + np.eye(num_features) * 1e-6
        #     else:
        #         self.covar[k] = np.eye(num_features)
        
        # # Initialize weights
        # self.weights = np.zeros(self.c)
        # for k in range(self.c):
        #     self.weights[k] = np.sum(labels == k) / n

        self.log_likelihood_prev = None
        self.log_likelihood = None

        for i in range(self.max_iterations):
            self.r = self.getMembership(X)
            self.weights, self.means, self.covar = self.getParams(X)  # update parameters
            self.log_likelihood = self.getLikelihood(X)
            
            if self.log_likelihood_prev is not None and abs(self.log_likelihood - self.log_likelihood_prev) < 1e-6:  # convergence check
                print(f'Convergence reached after {i} iterations')
                break

            self.log_likelihood_prev = self.log_likelihood
        

    def PDF_gaussians(self, X, mean, covar):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X = (X - self.mean) / self.std
        num_features = X.shape[1]
        regularized_covar = covar + np.eye(num_features) * 1e-4
        pdf = multivariate_normal(mean=mean, cov=regularized_covar, allow_singular=True).pdf(X)
        return pdf
    
        # self.mean = np.mean(X,axis=0)
        # self.std = np.std(X,axis=0) + 1e-8
        # X = (X - self.mean)/self.std # standardize
        # num_features = X.shape[1]

        # covar += np.eye(num_features) * 1e-4 # Regularize covar matrix to avoid singularity (chatgpt)

        # pdf = multivariate_normal(mean = mean, cov = covar, allow_singular=True).pdf(X)
        # # print("Printing PDF", pdf)
        # return pdf
    

        # num_features = X.shape[1]

        # covar += np.eye(num_features) * 1e-4 # Regularize covar matrix to avoid singularity (chatgpt)

        # covar_inv = np.linalg.inv(covar)
        # covar_det = np.linalg.det(covar)
        # diff = X - mean
        # exponent = -0.5 * np.sum(diff @ covar_inv * diff, axis=1)
        # coefficient = -0.5 * (num_features * np.log(2 * np.pi) + np.log(covar_det))
        # log_prob = coefficient + exponent
        # return np.exp(log_prob) # converting log probability to normal probability
    
    
    def getMembership(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0) + 1e-8
        X = (X - self.mean)/self.std # standardize
        num = np.zeros((X.shape[0], self.c))
        for c in range(self.c):
            # print(f'covariances: {self.covar}')
            # print(f'means: {self.means}')
            # print(f'weights: {self.weights}')
            pdf = self.PDF_gaussians(X, self.means[c], self.covar[c])
            # print(f'pdf: {pdf}')
            num[:, c] = self.weights[c] * pdf
        # print(f'num: {num}')
        total = np.sum(num, axis=1, keepdims=True)
        # print(f'total: {total}')
        # total = np.where(total == 0, 1e-10, total)  # Add a small value to avoid zero division
        total += 1e-6
        r = num / total
        return r
    

    def getParams(self, X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0) + 1e-8
        X = (X - self.mean)/self.std # standardize
        n, num_features = X.shape

        #weight update
        weights = np.sum(self.r, axis=0) / n

        # w_scaler = MinMax(self.weights)
        # w_scaler.scale()
        # print(f'weights: {self.weights}')

        #mean update
        means = np.dot(self.r.T, X) / (np.sum(self.r, axis=0)[:, np.newaxis]+1e-6)

        # m_scaler = MinMax(self.means)
        # m_scaler.scale()
        # print(f'means: {self.means}')

        cluster_responsibilities = np.sum(self.r, axis=0)  # Shape: (c,)
        cluster_responsibilities = cluster_responsibilities[:, np.newaxis]  # Shape: (c, 1)
        weighted_sums = np.dot(self.r.T, X)  # Shape: (c, num_features)
        means = weighted_sums / (cluster_responsibilities+1e-6)  # Shape: (c, num_features)

        #covar update
        covar = np.zeros((self.c, num_features, num_features))
        for c in range(self.c):
            diff = X - means[c]
            diff_T = diff.T
            rc_diff = self.r[:, c] * diff_T
            covar[c] = np.dot(rc_diff, diff) / (np.sum(self.r[:, c])+1e-6)
            # check_covariance(covar[c])
        # c_scaler = MinMax(self.covar)
        # c_scaler.scale()
        # print(f'covar: {self.covar}')

        self.weights = weights
        self.means = means
        self.covar = covar
        # self.covar = nearest_positive_definite(self.covar)
        # self.covar = clip_covariance_eigenvalues(self.covar, min_eigenvalue=1e-3)

        return self.weights, self.means, self.covar
    

    def getLikelihood(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0) + 1e-8
        X = (X - self.mean)/self.std # standardize
        likelihood = np.zeros((X.shape[0], self.c))
        for c in range(self.c):
            likelihood[:, c] = self.weights[c] * self.PDF_gaussians(X, self.means[c], self.covar[c])
        likelihood_total = np.sum(likelihood, axis=1) + 1e-10
        log_likelihood = np.sum(np.log(likelihood_total))
        # likelihood_total = np.sum(likelihood, axis=1)
        # return likelihood_total
        return log_likelihood








# import numpy as np

# class GMM2:

#     def __init__(self, c=2, max_iterations=100):
#         self.c = c
#         self.max_iterations = max_iterations
#         self.weights = None  # pi i.e. prior probabilities that a sample belongs to a particular cluster
#         self.means = None
#         self.covar = None
#         self.r = None  # one hot RV denoting cluster to which a sample belongs
#         self.log_likelihood = None


#     def fit(self, X):
#         X = X.to_numpy()
#         n, num_features = X.shape

#         self.weights = np.ones(self.c) / self.c # initialise all weights with 1/n so that sum=1
#         self.means = X[np.random.choice(n, self.c, replace=False)] # randomly take any value as mean from the 
#         self.covar = np.array([np.eye(num_features) for _ in range(self.c)]) #create an identity matrix of size num_features x num_features for each c.
#         self.log_likelihood = []
#         for i in range(self.max_iterations):
#             likelihood = np.zeros((n, self.c))

#             # calculate ric for each sample in the data
#             for c in range(self.c):
#                 likelihood[:, c] = self.weights[c] * self.PDF_gaussians(X, self.means[c], self.covar[c])
#             total_likelihood = np.sum(likelihood, axis=1, keepdims=True)
#             total_likelihood = np.where(total_likelihood == 0, 1e-6, total_likelihood)  # Add a small value to avoid zero division
#             # print(likelihood)
#             self.r = likelihood / total_likelihood
#             # self.r = self.getMembership(X)

#             # update params

#             #weight update
#             weights = np.sum(self.r, axis=0) / n
            
#             #mean update
#             means = np.dot(self.r.T, X) / np.sum(self.r, axis=0)[:, np.newaxis]

#             #covar update
#             covar = np.zeros((self.c, num_features, num_features))
#             for c in range(self.c):
#                 diff = X - means[c]
#                 weighted_diff = self.r[:, c][:, np.newaxis] * diff
#                 covar[c] = np.dot(weighted_diff.T, diff) / np.sum(self.r[:, c])

#             self.weights = weights
#             self.means = means
#             self.covar = covar

#             # # calculate log likelihood
#             # likelihood = np.zeros((X.shape[0], self.c))
#             # for c in range(self.c):
#             #     likelihood[:, c] = self.weights[c] * self.PDF_gaussians(X, self.means[c], self.covar[c])

#             log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
#             # log_likelihood = self.getLikelihood(X)
#             self.log_likelihood.append(log_likelihood)

#             #check convergence
#             if i > 0 and abs(self.log_likelihood[-1] - self.log_likelihood[-2]) < 1e-6:
#                 print(f'Convergence reached after {i} iterations') 
#                 break
        

#     def PDF_gaussians(self, X, mean, covar):
#         num_features = X.shape[1]

#         covar += np.eye(num_features) * 1e-4 # Regularize covar matrix to avoid singularity (chatgpt)

#         distr = multivariate_normal(mean = mean, cov = covar)
#         pdf = distr.pdf(X)
#         return pdf
#         # covar_inv = np.linalg.inv(covar)
#         # covar_det = np.linalg.det(covar)
#         # covar_det = np.clip(covar_det, 1e-6, None) #(chatgpt)
#         # diff = X - mean
#         # exponent = -0.5 * np.sum(diff @ covar_inv * diff, axis=1)
#         # coefficient = -0.5 * (num_features * np.log(2 * np.pi) + np.log(covar_det))
#         # log_prob = coefficient + exponent
#         # return np.exp(log_prob) # converting log probability to normal probability
    
    
#     def getMembership(self):
#         # likelihood = np.zeros((X.shape[0], self.c))
#         # for c in range(self.c):
#         #     likelihood[:, c] = self.weights[c] * self.PDF_gaussians(X, self.means[c], self.covar[c])
#         # total_likelihood = np.sum(likelihood, axis=1, keepdims=True)
#         # total_likelihood = np.where(total_likelihood == 0, 1e-10, total_likelihood)  # Add a small value to avoid zero division
#         # r = likelihood / total_likelihood
#         return self.r
    

#     def getParams(self):
#         # n, num_features = X.shape

#         # #weight update
#         # weights = np.sum(self.r, axis=0) / n
        
#         # #mean update
#         # means = np.dot(self.r.T, X) / np.sum(self.r, axis=0)[:, np.newaxis]

#         # #covar update
#         # covar = np.zeros((self.c, num_features, num_features))
#         # for c in range(self.c):
#         #     diff = X - means[c]
#         #     weighted_diff = self.r[:, c][:, np.newaxis] * diff
#         #     covar[c] = np.dot(weighted_diff.T, diff) / np.sum(self.r[:, c])

#         # self.weights = weights
#         # self.means = means
#         # self.covar = covar

#         return self.weights, self.means, self.covar
    

#     def getLikelihood(self):
#         # likelihood = np.zeros((X.shape[0], self.c))
#         # for c in range(self.c):
#         #     likelihood[:, c] = self.weights[c] * self.PDF_gaussians(X, self.means[c], self.covar[c])

#         # log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
#         total_log_likelihood = np.sum(self.log_likelihood)
#         total_likelihood = np.exp(total_log_likelihood)
#         return total_likelihood

#         # NOTE
#         # self.r.T: Transposes the responsibilities matrix, making it shape (c, n).
#         # np.dot(self.r.T, X): Computes the weighted sums of the data points for each cluster.
#         # np.sum(self.r, axis=0): Sums the responsibilities for each cluster.
#         # Division normalizes the sums to obtain the updated cluster means.


# # import numpy as np
# # from scipy.stats import multivariate_normal

# class GMM3:
#     def __init__(self, k, n_iter=100, cont_th=1e-3):
#         self.k = k
#         self.n_iter = n_iter
#         self.cont_th = cont_th
#         self.means = None
#         self.covariances = None
#         self.weights = None
#         self.responsibilities = None

#     def getParams(self):
#         return {"means": self.means,"covariances": self.covariances,"weights": self.weights}

#     def fit(self, X): # done
#         n_samples, n_features = X.shape
#         # Weights
#         self.weights = np.random.rand(self.k)
#         self.weights /= np.sum(self.weights)
#         # Means
#         self.means = []
#         feature_min = np.min(X,axis=0)
#         feature_max = np.max(X,axis=0)
#         # self.means = feature_min + (feature_max-feature_min)*np.random.rand(self.k)
#         for i in range(self.k):
#             self.means.append(feature_min+(feature_max-feature_min)*np.random.rand(n_features))
#         self.means = np.array(self.means)
#         # print("Means shape",self.means.shape)
#         # print(self.means)
#         # Covariances
#         self.covariances = []
#         for i in range(self.k):
#             temp_mat = np.random.randn(n_features,n_features)
#             temp_mat = np.dot(temp_mat,temp_mat.T)
#             self.covariances.append(temp_mat)
#         self.covariances = np.array(self.covariances)
#         print(self.covariances)
#         # self.means = X[np.random.choice(n_samples, self.k,replace=False)]
#         # print(self.means)
#         # self.covariances = np.array([np.eye(n_features)] * self.k)
#         # self.weights = np.ones(self.k) / self.k

#         log_likelihood_old = None
#         for iteration in range(self.n_iter):
#             # Expectation step
#             self.getMembership(X)

#             # Maximization step
#             self.MaximizationStep(X)
#             log_likelihood_new = self.getLikelihood(X)
#             if log_likelihood_old != None and abs(log_likelihood_new - log_likelihood_old) < self.cont_th:
#                 # print(f"Converges after {iteration + 1} iterations")
#                 break
#             log_likelihood_old = log_likelihood_new

#     def getMembership(self,X): # done
#         n_samples = X.shape[0]
#         self.responsibilities = np.zeros((n_samples, self.k))
#         # print("Printing shape in getMembership",X.shape)
#         for k in range(self.k):
#             print(self.means[k],self.covariances[k])
#             pdf = self.Gaussian(X, self.means[k], self.covariances[k])
#             # print("PDF Shape",pdf.shape)
#             self.responsibilities[:, k] = self.weights[k] * pdf
#             print("PDF\n",pdf)
#             print(f"Loop {k}",self.responsibilities[:,k])

#         # print("Responsibilities sum\n",self.responsibilities.sum(axis=1, keepdims=True))
#         self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)
#         return self.responsibilities

#     def getLikelihood(self, X): # done
#         log_likelihood = 0
#         for k in range(self.k):
#             pdf = self.Gaussian(X, self.means[k], self.covariances[k])
#             log_likelihood += self.weights[k] * pdf
#         return np.sum(np.log(log_likelihood))

#     def MaximizationStep(self, X): # done
#         n_samples,n_features = X.shape
#         temp = self.responsibilities.sum(axis=0)
#         self.weights = temp / n_samples
#         # print(np.dot(self.responsibilities.T, X))
#         # print(self.responsibilities)
#         covariances = np.zeros((self.k,n_features,n_features))
#         self.means = np.dot(self.responsibilities.T, X) / temp[:, np.newaxis]
#         for k in range(self.k):
#             X_centered = X - self.means[k]
#             covariances[k] = np.dot(self.responsibilities[:, k] * X_centered.T, X_centered) / temp[k]
#         self.covariances = covariances
#         print("Covariances\n",self.covariances)

#     def Gaussian(self, X, mean, covariance): # done
#         n_features = X.shape[1]
#         # covariance += np.eye(n_features) * 1e-4
#         # print("Mean\n",mean)
#         # print("Covariance\n",covariance)
#         return multivariate_normal(mean=mean,cov=covariance).pdf(X)
#         # print(covariance.shape)

#         # covariance_inv = np.linalg.inv(covariance)
#         # diff = X - mean # (x-\mu) matrix
#         # # Chatgpt (Starts here)
#         # exponent = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
#         # # Ends here

#         # covariance_det = np.linalg.det(covariance)
#         # covariance_det = np.clip(covariance_det,1e-10,None)
#         # # print(covariance_det)

#         # # norm_const = np.sqrt((2 * np.pi) ** n_features * covariance_det)
#         # log_norm_const = (n_features / 2) * np.log(2 * np.pi) + 0.5 * np.log(covariance_det)
#         # norm_const = np.exp(log_norm_const)