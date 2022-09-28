import numpy as np

from scipy.spatial import Delaunay

from class_mesh_simplify import MeshSimplify

def create_boundary_edges(N):

    boundary_edges = {}
    for i in range(1, N - 1):
        boundary_edges[i] = [i - 1, i + 1]
        boundary_edges[i * N] = [(i - 1)*N, (i + 1)*N]
        boundary_edges[(i + 1)*N - 1] = [i*N - 1, (i + 2)*N - 1]
        boundary_edges[N*(N - 1) + i] = [N*(N - 1) + i - 1, N*(N - 1) + i + 1]

    # add corners
    boundary_edges[0] = [1, N]
    boundary_edges[N - 1] = [N - 2, 2*N - 1]
    boundary_edges[N*(N - 1)] = [N*(N - 2), N*(N - 1) + 1]
    boundary_edges[N*N - 1] = [N*N - 2, N*(N - 1) - 1]
    
    return boundary_edges

def get_quadrics_points(F, N_init, N_final, threshold_prct=0.1):
    x_l, x_u = F.bounds[0]
    if len(F.bounds) == 1:
        y_l, y_u = x_l, x_u
    else:
        y_l, y_u = F.bounds[1]

    threshold = (x_u - x_l)/2. * threshold_prct # threshold_prct is the percentage of x_u - x_l we are willing to merge.  

    x = np.linspace(x_l, x_u, int(N_init**(1/2.)))
    y = np.linspace(y_l, y_u, int(N_init**(1/2.)))
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()

    #define 2D points, as input data for the Delaunay triangulation of U
    points2D=np.vstack([x, y]).T
    tri = Delaunay(points2D)#triangulate the rectangle U"
    z = F.f(points2D)

    simplify_ratio = N_final/float(N_init)

    # create boundary
    points = np.copy(np.array([x, y, z]).T)
    faces = np.copy(tri.simplices)
    boundary_edges = create_boundary_edges(int(N_init**(1/2.)))
    interior_v = np.array([(x_u - x_l)/2., (y_u - y_l)/2., F.f(np.array([[(x_u - x_l)/2., (y_u - y_l)/2.]]))[0]])

    # # Here, point and vertex are same terms
    # # Read 3d model, initialization (points/vertices, faces, edges), compute the Q matrices for all the initial vertices
    model=MeshSimplify(points, faces + 1, boundary_edges, interior_v, threshold, simplify_ratio)

    # # Select all valid pairs.
    model.generate_valid_pairs()

    # # Compute the optimal contraction target v_opt for each valid pair (v1, v2)
    # # The error v_opt.T*(Q1+Q2)*v_opt of this target vertex becomes the cost of contracting that pair.
    # # Place all the pairs in a heap keyed on cost with the minimum cost pair at the top
    model.calculate_optimal_contraction_pairs_and_cost()

    # # Iteratively remove the pair (v1, v2) of least cost from the heap
    # # contract this pair, and update the costs of all valid pairs involving (v1, v2).
    # # until existing points = ratio * original points
    model.iteratively_remove_least_cost_valid_pairs(verbose=True)

    # # Generate the simplified 3d model (points/vertices, faces)
    model.generate_new_3d_model()

    # # Output the model to output_filepath
    # model.output(output_filepath)

    new_points = model.points
    new_faces = model.faces - 1 # zero adjust
    return new_points, new_faces