from chart_studio import plotly as py
import plotly.graph_objs as go

import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay

from functools import reduce


def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value zval to a corresponding color in the colormap

    if vmin>vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val
    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'


def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices 
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z)).T
    tri_vertices= points3D[simplices] #map(lambda index: points3D[index], simplices)# vertices of the surface triangles     
    zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of 
                                                      #triangle vertices
    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)
    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    I,J,K=tri_indices(simplices)

    triangles=go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor,
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )

    if plot_edges is None:# the triangle sides are not plotted 
        return [triangles]
    else:
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]

        #define the lines to be plotted
        lines=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color= 'rgb(50,50,50)', width=1.5)
               )
        return [triangles, lines]



def tmp():
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Mesh3d(
        x = prmt[0],
        y = prmt[1],
        z = z
        )
    ])







    fig.show()



def get_full_plane_along_uv(F, u, v, N):
    c = np.mean(F.bounds, axis=1)
    consts = F.bounds[:, 1] - c
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    
    eps = 1e-5

    res = []
    for i in range(dim):
        for j in range(i+1, dim):
            A = np.array([[u[i], v[i]], [u[j], v[j]]])
            if np.abs(np.linalg.det(A)) < eps:
                continue
            ts10 = np.linalg.solve(A, np.array([1, 0]))
            ts01 = np.linalg.solve(A, np.array([0, 1]))
            vec10 = ts10[0] * u + ts10[1] * v
            vec01 = ts01[0] * u + ts01[1] * v
            
            if np.max(np.abs(vec10)) <= 1 + eps and np.max(np.abs(vec01)) <= 1 + eps:
                res.append([vec10, vec01])
                
    tlin = np.linspace(-1, 1, N)
    ts = np.meshgrid(tlin, tlin)
    t1 = ts[0].ravel()
    t2 = ts[1].ravel()

    return consts*(np.array([t1, t2]).T @ np.array(res[0])) + c
    
def get_constrained_plane_along_uv(F, u, v, N):
    """Plane goes through the center of the cube"""
    dim = len(u)
    bounds = F.bounds
    if len(bounds) != u.shape[0]:
        bounds = np.array([bounds[0] for _ in range(u.shape[0])])
    c = np.mean(bounds, axis=1)
    consts = bounds[:, 1] - c
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    
    eps = 1e-5

    res = []
    for i in range(dim):
        for j in range(i+1, dim):
            A = np.array([[u[i], v[i]], [u[j], v[j]]])
            if np.abs(np.linalg.det(A)) < eps:
                continue
            ts10 = np.linalg.solve(A, np.array([1, 0]))
            ts01 = np.linalg.solve(A, np.array([0, 1]))
            vec10 = ts10[0] * u + ts10[1] * v
            vec01 = ts01[0] * u + ts01[1] * v
            
            if np.max(np.abs(vec10)) <= 1 + eps and np.max(np.abs(vec01)) <= 1 + eps:
                res.append([[vec10, vec01], [i, j]])
                
    tlin = np.linspace(-1, 1, N)
    ts = np.meshgrid(tlin, tlin)
    t1 = ts[0].ravel()
    t2 = ts[1].ravel()
        
    good_ts = []
    for i in range(len(t1)):
        if np.max(np.abs(t1[i] * res[0][0][0] + t2[i] * res[0][0][1])) <= 1:
            good_ts.append(i)        
            
    filtered_ts = np.array([t1[good_ts], t2[good_ts]]).T

    pts = consts*(filtered_ts @ np.array(res[0][0])) + c
    
    prmtrztn = np.array([consts[res[0][1][0]] * np.linalg.norm(res[0][0][0]) * filtered_ts[:, 0],
                         consts[res[0][1][1]] * np.linalg.norm(res[0][0][1]) * filtered_ts[:, 1]])
    
    return prmtrztn, np.array(pts)

    
def get_line_along_v(F, v, N):
    """Goes through the center of the cube"""
    v = v / np.linalg.norm(v)
    c = np.mean(F.bounds, axis=1)
    # when does t*v + c cross boundary. 
    # bounds[:, 0] \leq c + t v \leq bounds[:, 1]
    # ts_low \leq t \leq ts_high. with ts_low \leq 0 and ts_high \geq 0. 
    # hence, of all negatives we need the largetst and all positives the smallest. 
    # because c is the center, we have t_low = -t_high

    t0s =  (F.bounds[:, 0] - c)/v
    t1s =  (F.bounds[:, 1] - c)/v
    t_idx = np.argmin(np.abs(t0s))
    
    t0 = t0s[t_idx]
    t1 = t1s[t_idx]
    
    xs = (c + t0 * v) + np.linspace(0, 1, N).reshape(-1, 1) * (c + t1 * v - (c + t0 * v))
    return np.array(xs)


def test_helper():
    bounds = F.bounds
    if len(bounds) != u.shape[0]:
        bounds = np.array([bounds[0] for _ in range(u.shape[0])])

    prmt, interior = get_constrained_plane_along_uv(F, u, v, N=250)
    
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=np.array(np.array([-1, -1, 1, 1, -1, -1, 1, 1])*max(bounds[0])),
            y=np.array(np.array([-1, 1, 1, -1, -1, 1, 1, -1])*max(bounds[1])),
            z=np.array(np.array([-1, -1, -1, -1, 1, 1, 1, 1])*max(bounds[2])),
            colorbar_title='z',
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.linspace(0, 1, 12, endpoint=True),
            intensitymode='cell',
            # i, j and k give the vertices of triangles
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            name='y',
            showscale=True,opacity=0.23
        ), 
        go.Mesh3d(
        x = interior[:, 0],
        y = interior[:, 1],
        z = interior[:, 2]
        )
    ])

    fig.show()