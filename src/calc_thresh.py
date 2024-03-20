# import numpy as np


# class Threshold:
#     def calc_dist_mean_emb(self, mean_embeddings, val_embeddings):
#         mean = np.array(mean_embeddings)
#         val = np.array(val_embeddings)
#         dist = np.linalg.norm(mean - val)
#         return dist

#     def calc_e_sc(self, dist_list):
#         esc = sum(dist_list) // len(dist_list)
#         return esc

#     def calc_e_dc(self, dist_list):
#         edc = sum(dist_list) // len(dist_list)
#         return edc

#     def calc_individual_thresh(self, esc, edc):
#         individual_thresh = (esc + edc) // 2
#         return individual_thresh

#     def calc_overall_thresh(self, individual_thresh_list):
#         overal_thresh = sum(individual_thresh_list) // len(individual_thresh_list)
#         return overal_thresh

#     def main(self, mean_data, val_embeddings):
#         i_thresh = []
#         e_sc_list = []
#         e_dc_list = []
#         for mean_embedding, name in zip(mean_data[0], mean_data[1]):
#             for val_embedding, val_name in zip(val_embeddings[0], val_embeddings[1]):
#                 dist = self.calc_dist_mean_emb(mean_embedding, val_embedding)
#                 if name == val_name:
#                     e_sc_list.append(dist)
#                 else:
#                     e_dc_list.append(dist)
#             e_sc = self.calc_e_sc(e_sc_list)
#             e_dc = self.calc_e_dc(e_dc_list)
#             i_thresh.append(self.calc_individual_thresh(e_sc, e_dc))
#         threshold = self.calc_overall_thresh(i_thresh)
#         return threshold

import numpy as np

class Threshold:
    
    def __init__(self, mean_data=None, val_embeddings=None):
        self.mean_data = mean_data
        self.val_embeddings = val_embeddings
    
    def __call__(self):
        i_thresh = []
        for mean_embedding, name in zip(self.mean_data[0], self.mean_data[1]):
            e_sc_list = []
            e_dc_list = []
            for val_embedding, val_name in zip(self.val_embeddings[0], self.val_embeddings[1]):
                dist = self.calc_dist_mean_emb(mean_embedding, val_embedding)
                if name == val_name:
                    e_sc_list.append(dist)
                else:
                    e_dc_list.append(dist)
            e_sc = self.calc_e_sc(e_sc_list)
            e_dc = self.calc_e_dc(e_dc_list)
            i_thresh.append(self.calc_individual_thresh(e_sc, e_dc))
        threshold = self.calc_overall_thresh(i_thresh)
        return threshold
    
    def calc_dist_mean_emb(self, mean_embeddings, val_embeddings):
        mean = np.array(mean_embeddings)
        val = np.array(val_embeddings)
        dist = np.linalg.norm(mean - val)
        return dist
    
    def calc_e_sc(self, dist_list):
        return sum(dist_list) / len(dist_list)

    def calc_e_dc(self, dist_list):
        return sum(dist_list) / len(dist_list)

    def calc_individual_thresh(self, esc, edc):
        return (esc + edc) // 2

    def calc_overall_thresh(self, individual_thresh_list):
        return sum(individual_thresh_list) // len(individual_thresh_list)

