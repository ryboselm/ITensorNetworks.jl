using ITensorNetworks: 
    contract, TTN, AbstractITensorNetwork, ITensorNetwork, AbstractTTN, TTN
using ITensors: inds, ITensor, svd, random_itensor
using ITensors
using Graphs: SimpleGraph, path_graph, binary_tree
using NamedGraphs.GraphsExtensions: subgraph
using LinearAlgebra

# Reference: https://arxiv.org/pdf/2209.03196 
# (Automatic structural optimization of tree tensor networks)

function structure_sweep(tn::AbstractTTN)

    # assert that tn is a binary tree
    @assert is_binary(tn)

    n_tensors = length(tn.tensors)
    graph = SimpleGraph(n_tensors)

    # start with initial TTN layout
    first_tensor = tn[1]
    # needs to start with a center somewhere. pick something on the edge of the tree. replace this itensor with its SVD inside the ITensorNetwork.
    # there should be three connected tensors where there used to be just one tensor. Maintain something that tracks what the current diagonal tensor is.
    # take subgraph of the nodes adjacent to it. pass to local_structure_update()
    # this does some funky re-arrangement of the tensors and creates new indices.
    
    # repeat to sweep through the whole tree
    # do multiple sweeps of the whole tree

    return not_implemented()
end

function local_structure_update(tn::AbstractITensorNetwork)
    # verify that the structure of tn is correct
    ψ = contract(tn)
    indz = inds(ψ)
    @assert length(indz) == 4 "Expected 4 indices, got $(length(indz))"

    left_inds_configurations = [
        (indz[1],indz[2]), 
        (indz[1],indz[3]), 
        (indz[1],indz[4])
    ]

    min_EE_choice = 0
    min_EE = Inf
    for (i,left_inds) in enumerate(left_inds_configurations)
        U, S, V = ITensors.svd(ψ, left_inds...)
        EE = entanglement_entropy(S)
        if EE < min_EE
            min_EE_choice, min_EE = i, EE
        end
    end

    U, S, V = ITensors.svd(ψ, left_inds_configurations[min_EE_choice]...)
    return ITensorNetwork([U, S, V]) # return a new tensor network subgraph
end

function entanglement_entropy(T::ITensor)
    ind1, ind2 = inds(T)
    @assert dim(ind1) == dim(ind2) "2D diagonal tensor expected"
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

# checks if a TTN is a binary tree
function is_binary(tn::ITensorNetwork)
    return true
end

tn = ITensorNetwork(binary_tree(3); link_space=2)
for (v,tensor) in enumerate(tn)
    tn[v] = random_itensor(inds(tensor)...)
end
sub_tn = subgraph(v -> v < 4 , tn)
@show sub_tn
tn2 = local_structure_update(sub_tn)
@show tn2
for i in 1:3
    @show tn2[i]
end


