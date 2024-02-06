import secrets 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def categorical_distance(row1, row2):
    return np.sum(row1 != row2)

class MDAVGeneric:
    '''
    Implements the MDAV-generic algorithm from [1].

    TODO: maybe add improvements from [2]?

    [1]: Domingo-Ferrer, J., & Torra, V. (2005). Ordinal, continuous and heterogeneous 
    k-anonymity through microaggregation. Data Mining and Knowledge Discovery, 11, 195–212.

    [2]: Rodríguez-Hoyos, A., Estrada-Jiménez, J., Rebollo-Monedero, D., Mezher, A. M., 
    Parra-Arnau, J., & Forné, J. (2020). The Fast Maximum Distance to Average Vector 
    (F-MDAV): An algorithm for k-anonymous microaggregation in big data. Engineering 
    Applications of Artificial Intelligence, 90, 103531. 
    '''
    def __init__(self, df, feature_columns):
        self.df = df
        self.feature_columns = feature_columns
        self.num_data_points = len(self.df.index)

    def partition(self, k):
        self.__prepare_data()
        self.__build_distance_matrix()
        self.clusters = []

        while (len(self.remaining_indices) >= 3 * k):
            centroid = self.__find_centroid()
            far_end_dist = self.__get_distance_vector_of_most_distant_point(centroid)
            self.__assign_closest_points_to_new_cluster(far_end_dist, k)

            other_end_dist = self.__get_distance_vector_of_most_distant_point_from_data_point(far_end_dist.loc[self.remaining_indices])
            self.__assign_closest_points_to_new_cluster(other_end_dist, k)

        if len(self.remaining_indices) >= 2 * k:
            centroid = self.__find_centroid()
            far_end_dist = self.__get_distance_vector_of_most_distant_point(centroid)
            self.__assign_closest_points_to_new_cluster(far_end_dist, k)

        self.clusters.append(self.remaining_indices)

        return self.clusters 

    def __prepare_data(self):
        self.remaining = self.df.drop(columns=[column for column in self.df.columns if column not in self.feature_columns])
        self.remaining_indices = self.remaining.index
        self.categorical = [column for column in self.feature_columns if self.remaining[column].dtype == "category"]
        self.continuous  = [column for column in self.feature_columns if self.remaining[column].dtype != "category"]
       
        self.remaining.loc[:, self.continuous] = (self.remaining.loc[:, self.continuous] - self.remaining.loc[:, self.continuous].mean()) / self.remaining.loc[:, self.continuous].std()

        for attribute in self.categorical:
            # self.categorical.loc[:, attribute] = pd.factorize(self.categorical.loc[:, attribute])[0]
            self.remaining.loc[:, attribute] = pd.factorize(self.remaining.loc[:, attribute])[0]

    def __find_centroid(self):
        centroid_continuous = self.remaining.loc[self.remaining_indices, self.continuous].mean()
        modes = self.remaining.loc[self.remaining_indices, self.categorical].mode()
        centroid_categorical = [secrets.choice(modes.loc[:, column].dropna()) for column in modes]
        centroid_categorical = np.array(centroid_categorical)
        return (centroid_continuous, centroid_categorical)

    def __get_distance_vector_of_most_distant_point(self, position):
        pos_continuous, pos_categorical = position
        distances1 = cdist([pos_continuous], self.remaining.loc[self.remaining_indices, self.continuous], metric='euclidean')[0]
        distances2 = cdist([pos_categorical], self.remaining.loc[self.remaining_indices, self.categorical], metric='hamming')[0] * len(self.categorical)
        distances = distances1 + distances2
        max_dist_array_pos = distances.argmax()
        max_dist_index = self.remaining_indices[max_dist_array_pos]
        return self.distance_matrix.loc[max_dist_index, self.remaining_indices]

    def __get_distance_vector_of_most_distant_point_from_data_point(self, distance_vector):
        max_dist_array_pos = distance_vector.argmax()
        max_dist_index = self.remaining_indices[max_dist_array_pos]
        return self.distance_matrix.loc[max_dist_index, self.remaining_indices]

    def __assign_closest_points_to_new_cluster(self, distance_vector, k):
        array_positions = np.argpartition(distance_vector, k)[:k]

        indices = self.remaining_indices.take(array_positions)
        self.remaining_indices = self.remaining_indices.drop(indices)

        self.clusters.append(indices)

    def __build_distance_matrix(self):
        dist_continuous = pdist(self.remaining.loc[:, self.continuous], metric='euclidean')
        dist_categorical = (pdist(self.remaining.loc[:, self.categorical], metric='hamming') * len(self.categorical)) if not len(self.categorical) == 0 else 0

        self.distance_matrix = pd.DataFrame(squareform(dist_categorical + dist_continuous))



class RandomChoiceAggregation:
    '''
    Implements the microaggregation algorithm originally used
    in the k-Same family of algorithms [1]. Only designed for
    numerical data.
    
    The algorithm selects a random data point and assigns its
    k-1 closest neighbours to a new cluster.

    [1]: E. M. Newton, L. Sweeney, and B. Malin, ‘Preserving privacy
    by de-identifying face images’, IEEE Transactions on Knowledge 
    and Data Engineering, vol. 17, no. 2, pp. 232–243, Feb. 2005, 
    doi: 10.1109/TKDE.2005.32.

    '''
    def __init__(self, df, feature_columns):
        self.df = df
        self.feature_columns = feature_columns
        self.num_data_points = len(self.df.index)

    def partition(self, k):
        self.__prepare_data()
        self.clusters = []

        while (len(self.remaining_indices) >= 2 * k):
            # choose random point
            # assign closest points to new cluster
            point = self.remaining.loc[self.remaining_indices].sample(n=1)
            self.__assign_closest_points_to_new_cluster(point, k)
            
        self.clusters.append(self.remaining_indices)

        return self.clusters

    def __prepare_data(self):
        self.remaining = self.df.drop(columns=[column for column in self.df.columns if column not in self.feature_columns])
        self.remaining_indices = self.remaining.index

    def __assign_closest_points_to_new_cluster(self, point, k):
        distance_vector = cdist(point, self.remaining.loc[self.remaining_indices], metric='euclidean')[0]
        array_positions = np.argpartition(distance_vector, k)[:k]
        indices = self.remaining_indices.take(array_positions)
        self.remaining_indices = self.remaining_indices.drop(indices)

        self.clusters.append(indices)
