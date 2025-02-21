using ITensorNetworks: 
    TTN, AbstractITensorNetwork, ITensorNetwork, siteinds, random_tensornetwork, is_tree, 
    tensors, AbstractTreeTensorNetwork, TTN, AbstractTTN, tree_orthogonalize, orthogonalize
using ITensors: inds, ITensor, svd, random_itensor, Index, contract
using ITensors
using Graphs: SimpleGraph, path_graph, binary_tree, edges, Edge
using NamedGraphs: NamedEdge, src, dst
using NamedGraphs.GraphsExtensions: subgraph
using LinearAlgebra
using Random

# Reference: https://arxiv.org/pdf/2209.03196 
# (Automatic structural optimization of tree tensor networks)

"""
    struture_sweep()
    - tn: Tree Tensor Network to be optimized. (For now) each tensor must have 3 indices as was the case in the paper.
    - n_sweeps: number of times a full tree sweep is repeated

    Implements an automatic structural optimization procedure for tree tensor networks.
"""
function structure_sweep(tn::AbstractTTN; n_sweeps=1)
    # assert that each tensor in the tree is order 3
    @assert is_valid_tree_format(tn) "input tree does not have three edges at each node!"

    n_tensors = length(tensors(tn))

    # start with initial TTN layout, orthogonalize somewhere in the middle
    center = n_tensors ÷ 2
    tn = tree_orthogonalize(tn, center)
    tn = svd_tensor_network(tn, center)

   
    # there should be three connected tensors where there used to be just one tensor. Maintain something that tracks what the current diagonal tensor is.
    # take subgraph of the nodes adjacent to it. pass to local_structure_update()
    # this does some funky re-arrangement of the tensors and creates new indices.
    
    # repeat to sweep through the whole tree
    # do multiple sweeps of the whole tree

    return tn
end

# local update step of structural optimizer. Re-arranges indices according to local minimal entanglement entropy heuristic.
function local_structure_update!(tn::AbstractTTN, central_bond::NamedEdge, off_central_bond::NamedEdge)
    # Get the three vertices in order (unshared1, shared, unshared2)
    v1, v2, v3 = get_three_vertices(central_bond, off_central_bond)
    
    # Store original tensors before contraction
    t1, t2, t3 = tn[v1], tn[v2], tn[v3]
    
    # Contract the tensors to get the 4-index tensor
    ψ = contract(t1, t2, t3)
    indz = inds(ψ)
    @assert length(indz) == 4 "Expected 4 indices, got $(length(indz))"

    left_inds_configurations = [
        (indz[1],indz[2]), 
        (indz[1],indz[3]), 
        (indz[1],indz[4])
    ]

    # find optimal bipartition
    optimal_U = nothing
    optimal_S = nothing
    optimal_V = nothing
    min_EE = Inf
    for left_inds in left_inds_configurations
        U, S, V = ITensors.svd(ψ, left_inds...)
        EE = entanglement_entropy(S)
        if EE < min_EE
            min_EE = EE
            optimal_U = U
            optimal_S = S
            optimal_V = V
        end
    end

    tn[v1] = optimal_U
    tn[v2] = optimal_S
    tn[v3] = optimal_V

    return v2 # position of new center bond
end

# helper function for local_structure_update
function get_three_vertices(central_bond::NamedEdge, off_central_bond::NamedEdge)
    edge1_vertices = Set([src(central_bond), dst(central_bond)])
    edge2_vertices = Set([src(off_central_bond), dst(off_central_bond)])
    
    # Find shared vertex
    shared = intersect(edge1_vertices, edge2_vertices)
    @assert length(shared) == 1 "Edges must share exactly one vertex"
    shared_vertex = first(shared)
    
    # Find unshared vertices
    unshared = setdiff(union(edge1_vertices, edge2_vertices), shared)
    @assert length(unshared) == 2 "Expected two unshared vertices"
    
    # Get vertices from first edge excluding shared vertex
    unshared1 = first(setdiff(edge1_vertices, shared))
    # Get vertices from second edge excluding shared vertex  
    unshared2 = first(setdiff(edge2_vertices, shared))
    
    return (unshared1, shared_vertex, unshared2)
end

# should only be used on a diagonal matrix of singular values
function entanglement_entropy(T::ITensor)
    ind1, ind2 = inds(T)
    @assert dim(ind1) == dim(ind2) "2D diagonal matrix expected"
    EE = 0.0
    for i in 1:dim(ind1)
        D_i = T[i,i]^2
        if D_i > 0
            EE -= D_i * log(D_i)
        else
            @warn("D_i is zero and should be treated as zero.")
            EE += Inf
        end
    end
    return EE
end

# checks if each node of the TTN has three edges each
function is_valid_tree_format(tn::AbstractTTN)
    for tensor in tn
        num_indices = length(inds(tensor))
        if num_indices != 3
            return false
        end
    end
    return true
end

function svd_tensor_network(tn::AbstractTTN, node_idx::Int)
    # Get the tensor to split
    A = tn[node_idx]

    # Choose the split indices: assume the first half go to U, and the rest go to V
    inds_A = inds(A)
    @show inds_A
    num_inds = length(inds_A)
    split_point = num_inds ÷ 2

    # Define the left and right indices for SVD
    left_inds = inds_A[1:split_point]
    right_inds = inds_A[split_point+1:end]

    # Perform SVD
    U, S, V = svd(A, left_inds)

    tn[node_idx] = U
    tn[num_inds + 1] = S
    tn[node_idx + 1] = V

    return tn
end

"""
    create_rand_MPN(rng, N, χ)
    - rng: source of randomness
    - N: number of tensors
    - χ: internal bond dimension

    creates a random matrix product network initial state appropriate for the structural optimization algorithm
"""
function create_rand_MPN(rng::AbstractRNG, N::Int, χ::Int)
    @assert N ≥ 2 "N must be at least 2"

    s = siteinds("S=1/2", N+2)
    link_indices = [Index(χ, "link$i") for i in 1:(N-1)]
    A_1 = random_itensor(rng, s[1], s[2], link_indices[1])
    tensors = [A_1]
    for i in 2:N-1
        push!(tensors, random_itensor(rng, s[i+1], link_indices[i-1], link_indices[i]))
    end
    A_N = random_itensor(rng, s[N+1], s[N+2], link_indices[N-1])
    push!(tensors, A_N)
    tn = ITensorNetwork(tensors)
    return TTN(tn)
end


rng = MersenneTwister(1234)
tn = create_rand_MPN(rng, 5, 2)
tnx = structure_sweep(tn)
