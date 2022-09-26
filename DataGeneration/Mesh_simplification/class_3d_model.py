# -*- coding: utf-8 -*-
"""
@author: Anton Wang
"""

# 3D model class

import numpy as np
import scipy

class a_3d_model:
    def __init__(self, points, faces, boundary_edges, interior_v):
        self.points = points
        self.faces = faces
        self.boundary_edges = boundary_edges # self.boundary_edges[v1] is a list of neighbors of boundary vertex v1 which are also boundary verticies. 

        self.interior_v = interior_v
        self.penalty = 1000
        self.init_mesh()
        self.calculate_plane_equations()
        self.calculate_Q_matrices()
        
    def init_mesh(self):

        self.number_of_points=self.points.shape[0]
        self.number_of_faces=self.faces.shape[0]
        edge_1=self.faces[:,0:2]
        edge_2=self.faces[:,1:]
        edge_3=np.concatenate([self.faces[:,:1], self.faces[:,-1:]], axis=1)
        self.edges=np.concatenate([edge_1, edge_2, edge_3], axis=0)
        unique_edges_trans, unique_edges_locs=np.unique(self.edges[:,0]*(10**10)+self.edges[:,1], return_index=True)
        self.edges=self.edges[unique_edges_locs,:]
    
    def calculate_plane_equations(self):
        self.plane_equ_para = []
        for i in range(0, self.number_of_faces):
            # solving equation ax+by+cz+d=0, a^2+b^2+c^2=1
            # set d=-1, give three points (x1, y1 ,z1), (x2, y2, z2), (x3, y3, z3)
            p1=self.points[self.faces[i,0]-1, :]
            p2=self.points[self.faces[i,1]-1, :]
            p3=self.points[self.faces[i,2]-1, :]
            out_plane = calculate_plane_equation_for_one_face(p1, p2, p3)
            self.plane_equ_para.append(out_plane)
        self.plane_equ_para=np.array(self.plane_equ_para)
            
    def calculate_Q_matrices(self):
        self.Q_matrices = []
        for i in range(0, self.number_of_points):
            point_index=i+1
            # each point is the solution of the intersection of a set of planes
            # find the planes for point_index
            face_set_index=np.where(self.faces==point_index)[0]
            Q_temp=np.zeros((4,4))
            for j in face_set_index:
                p=self.plane_equ_para[j,:]
                p=p.reshape(1, len(p))
                Q_temp=Q_temp+np.matmul(p.T, p)

            if i in self.boundary_edges:
                for j in self.boundary_edges[i]:
                    edge_vec = self.points[i] - self.points[j]
                    normal = np.array([edge_vec[1], -edge_vec[0], 0]) # orthogonal to edge_vec and [0, 0, 1]
                    p = np.concatenate([normal, [-self.points[i] @ normal]])/np.linalg.norm(normal[:2])
                    Q_temp=Q_temp+self.penalty*np.outer(p, p)

            self.Q_matrices.append(Q_temp)

def calculate_plane_equation_for_one_face(p1, p2, p3):
    # input: p1, p2, p3 numpy.array, shape: (3, 1) or (1,3) or (3, )
    # p1 ,p2, p3 (x, y, z) are three points on a face
    # plane equ: ax+by+cz+d=0 a^2+b^2+c^2=1
    # return: numpy.array (a, b, c, d), shape: (1,4)
    p1=np.array(p1).reshape(3)
    p2=np.array(p2).reshape(3)
    p3=np.array(p3).reshape(3)
    point_mat=np.array([p1, p2, p3])
    if np.linalg.det(point_mat) == 0:
        abc = scipy.linalg.null_space(point_mat).T[0]
        output=np.concatenate([abc, [- abc @ p1]])/(np.sum(abc**2)**0.5)
    else:
        abc=np.matmul(np.linalg.inv(point_mat), np.array([[1],[1],[1]]))
        output=np.concatenate([abc.T, np.array(-1).reshape(1, 1)], axis=1)/(np.sum(abc**2)**0.5)
    output=output.reshape(4)
    return output