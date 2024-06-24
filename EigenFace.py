import matplotlib.pyplot as plt
import numpy as np
import scipy 
import cv2
import os
import pandas as pd


class EigenFaces:   
    
    def __init__(self, mode, training_data_filename=None, preloaded_training_data=None, test_data=None):
        self.mode=mode
        if mode== 'test' and training_data_filename is not None:
            preloaded_training_data = np.load(training_data_filename, allow_pickle=True)
        
        if preloaded_training_data is not None:
            self.training_images  = preloaded_training_data['training_images']
            self.n_training_images=len( preloaded_training_data['training_images']) 
            self.eigenfaces  = preloaded_training_data['eigenfaces']
            self.mean_face= preloaded_training_data['meanface']
            self.eigencoefficients = preloaded_training_data['eigencoefficients']
            self.eigenvalues= preloaded_training_data['eigenvalues']
            self.training_image_ids= preloaded_training_data['training_img_ids']
            self.image_height = self.training_images[0].shape[0] 
            self.image_width = self.training_images[0].shape[1] 
        if test_data is not None:
            self.test_images = test_data['test_images']
            self.test_eigencoefficients = test_data['test_eigencoefficients']
            self.n_test_images= len(test_data['test_images'])
            self.test_image_ids= test_data['test_img_ids']
    
    def read_images(self,directory):
        images = []
        image_ids = []
        for i in os.listdir(directory):
            if i.endswith('.pgm'):
                images.append(cv2.imread(directory+i, cv2.IMREAD_GRAYSCALE))
                image_ids.append(i[0:5])
                self.image_height = images[0].shape[0]
                self.image_width = images[0].shape[1]
        if self.mode=='train':
            self.training_images=images
            self.mean_face = self.get_mean_face(self.training_images)
            self.n_training_images=len(images)
            self.training_image_ids= image_ids
        #return self.training_images
        if self.mode=='test':    
            self.test_images=images
            self.n_test_images=len(images)
            self.test_image_ids= image_ids
        #return self.test_images

    
    
    def get_mean_face(self,images):
        return np.round(np.sum(images,axis=0)/len(images)).astype(int)
    
    def center_face(self,images):
        centered_data = images- self.mean_face
        return centered_data 

    def get_A_matrix(self):
        return np.dstack(self.centered_data)

    def flatten_A(self,A):
        A_flat = A.reshape(self.image_height*self.image_width,self.n_training_images)
        A_T_flat= A_flat.T
        return A_flat, A_T_flat

    
    def get_descending_eigs(self,S):
        # input: S is a symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        return eigenvalues, eigenvectors
    
    
    def find_eigenfaces(self):
        self.centered_training_images = self.center_face(self.training_images)
        A = np.dstack(self.centered_training_images)
        A_flattened, A_T_flattened = self.flatten_A(A)
        self.cov_matrix= (A_T_flattened @ A_flattened) / A.shape[2]
        self.eigenvalues, eigenvectors= self.get_descending_eigs(self.cov_matrix)
        # convert A^TA eigenvectors to AA^T eigenvectors and normalize
        self.eigenfaces = (A_flattened @ eigenvectors)  / np.linalg.norm((A_flattened @ eigenvectors), axis=0)


    def check_symmetric(self,A):
        return (A==A.T).all()

    def get_eigen_coefficients(self, thresh=None):
        y_i = []
        n = self.image_height*self.image_width
        
        if self.mode=='train':
            for i in range(self.n_training_images):
                y_i.append(np.array([self.centered_training_images[i].reshape(n) @ self.eigenfaces[:,j] for j in range(self.n_training_images)]))
            self.eigencoefficients = np.array(y_i)
        if self.mode=='test':
            self.centered_test_images = self.center_face(self.test_images)
            
            if thresh is not None:
                self.get_k_for_threshold(thresh)
                for i in range(self.n_test_images):
                    y_i.append(np.array([self.centered_test_images[i].reshape(n) @ self.eigenfaces[:,j] for j in range(self.k)]))
                self.test_eigencoefficients = np.array(y_i)
            else:    
                for i in range(self.n_test_images):
                    y_i.append(np.array([self.centered_test_images[i].reshape(n) @ self.eigenfaces[:,j] for j in range(self.n_training_images)]))
            self.test_eigencoefficients = np.array(y_i)
            

    def calc_mahalanobis_distance(self, image_coeffs, rank=1, thresh=None):
        if thresh is None: #use all eigencoefficients
            distances=[np.sum((1/self.eigenvalues)*(image_coeffs - self.eigencoefficients[i])**2) for i in range(self.n_training_images)]
        else: #use only top k eigencoefficients
            distances=[np.sum((1/self.eigenvalues[0:self.k])*(image_coeffs[0:self.k] - self.eigencoefficients[i][0:self.k])**2) for i in range(self.n_training_images)]
        idx = np.array(distances).argsort().astype(int)
        min_error = np.array(distances)[idx[:rank]]
        return idx[:rank], min_error
    
    def recognize_faces(self, rank=1, check_matches = 1,  n_faces='all', thresh=None):
        test_image_id = []
        matching_image = []
        distance_error=[]
        test_idx = []
        match_idx =[]
        if n_faces =='all':
            n_faces = np.arange(0,self.n_test_images)
        if thresh==None:
            for i in n_faces:
                idx, min_error = self.calc_mahalanobis_distance(self.test_eigencoefficients[i], rank=rank)
                test_image_id.append(self.test_image_ids[i])
                test_idx.append(i)
                match_idx.append(idx)
                matching_image.append(np.array(self.training_image_ids)[idx])
                distance_error.append(min_error)
        else:
            self.get_k_for_threshold(thresh)
            for i in n_faces:
                idx, min_error = self.calc_mahalanobis_distance(self.test_eigencoefficients[i], thresh=self.k, rank=rank)
                test_image_id.append(self.test_image_ids[i])
                matching_image.append(np.array(self.training_image_ids)[idx])
                test_idx.append(i)
                match_idx.append(idx)
                distance_error.append(min_error)
                
        classificationdf =  pd.DataFrame([test_image_id, test_idx, matching_image, match_idx, distance_error]).T.rename(columns={0:'query_id', 1: 'query_idx', 2:'best_match', 3:'match_idx', 4:'dist_error'} )
        
        
        for i in check_matches:
            classificationdf['isMatch_r'+str(i)] = [d in l[0:i] for d, l in zip(classificationdf['query_id'], classificationdf['best_match'])]

        self.classifications = classificationdf
        return classificationdf
    
    def get_k_for_threshold(self,thresh):## determine k eigenvectors to keep
        n = self.n_training_images
        self.k = np.where(np.array([(np.sum(self.eigenvalues[0:i]) > np.sum(self.eigenvalues) * thresh) for i in range(n)])==True)[-1][0]
        ## determine k eigenvectors to keep
    
    def save_data(self, filename):
        if self.mode=='train':
            trainingdata = np.array([self.training_images, self.eigenfaces, self.mean_face, self.eigencoefficients, self.eigenvalues, self.training_image_ids], dtype='object')
            datatosave = dict(zip(['training_images','eigenfaces','meanface','eigencoefficients', 'eigenvalues', 'training_img_ids'], trainingdata))
            np.savez(filename, **datatosave)
            print('data saved as file %s.npz' % filename)
        if self.mode=='test':
            testdata = [self.test_images, self.test_image_ids,  self.test_eigencoefficients,]
            testdatatosave = dict(zip(['test_images', 'test_img_ids', 'test_eigencoefficients'], testdata))
            np.savez(filename, **testdatatosave)
            print('data saved as file %s.npz' % filename)
     
    
    def get_intruder_info(self, directory, n_remove):
        image_ids = []
        intruders=[]
        intruder_ids=[]
        
        for i in os.listdir(directory):
            image_ids.append(i[0:5])

        unique_images_ids_sorted = np.unique(np.sort(image_ids))[:n_remove]        

        for j,i in enumerate(os.listdir(directory)):
            if np.any(i[0:5] == unique_images_ids_sorted):
                intruders.append(cv2.imread(directory+i, cv2.IMREAD_GRAYSCALE))
                intruder_ids.append(i[0:5])
        self.intruders= intruders
        self.intruder_image_ids = intruder_ids

    def get_ROC_error_info(self):
        self.classifications['dist_error_normed'] = self.classifications['dist_error'] /self.classifications.dist_error.max()
        thresholds= np.linspace(0,1,50)
        for i,j in enumerate(thresholds):
            self.classifications['Tr'+str(np.round(j,2))] = self.classifications['dist_error_normed'] < j

        intruder_df =self.classifications[self.classifications.query_id.isin(self.intruder_image_ids )]
        non_intruder_df = self.classifications[~self.classifications.query_id.isin(self.intruder_image_ids )]
        TP=[]
        FP=[]
        for i,j in enumerate(thresholds):
            TP.append(np.sum(non_intruder_df['Tr'+str(np.round(j,2))]))
            FP.append(np.sum(intruder_df['Tr'+str(np.round(j,2))]))

        self.true_positive_rate=np.array(TP)/len(non_intruder_df)
        self.false_positive_rate=np.array(FP)/len(intruder_df)
        self.ROC_thresholds=thresholds 
        


