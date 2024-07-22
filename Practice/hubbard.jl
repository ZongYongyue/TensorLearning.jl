#one band Hubbard model
function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,SU2Irrep,U1Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×SU₂×U₁]$(parts)"
end

abstract type Simulation end
name(s::Simulation) = string(typeof(s))

struct OB_Sim <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Float64
    P::Int64
    Q::Int64
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function OB_Sim(t, u, μ=0.0, P=1, Q=1, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, P, Q, svalue, bond_dim, period, kwargs)
    end
end
name(::OB_Sim) = "OB"


###############
# Hamiltonian #
###############

function SymSpace(P,Q,spin)
    if spin
        I = fℤ₂ ⊠ U1Irrep ⊠ U1Irrep
        Ps = Vect[I]((0, 0, -P) => 1, (0, 0, 2*Q-P) => 1, (1, 1, Q-P) => 1, (1, -1, Q-P) => 1)
    else
        I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
        Ps = Vect[I]((0, 0, -P) => 1, (0, 0, 2*Q-P) => 1, (1, 1 // 2, Q-P) => 1)
    end

    return I, Ps
end

function Hopping(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        Vup = Vect[I]((1, 1, Q) => 1)
        Vdown = Vect[I]((1, -1, Q) => 1)
    
        c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vup)
        blocks(c⁺u)[I((1, 1, Q-P))] .= 1
        blocks(c⁺u)[I((0, 0, 2*Q-P))] .= 1
        cu = TensorMap(zeros, ComplexF64, Vup ⊗ Ps ← Ps)
        blocks(cu)[I((1, 1, Q-P))] .= 1
        blocks(cu)[I((0, 0, 2*Q-P))] .= 1
        
        c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vdown)
        blocks(c⁺d)[I((1, -1, Q-P))] .= 1
        blocks(c⁺d)[I((0, 0, 2*Q-P))] .= 1
        cd = TensorMap(zeros, ComplexF64, Vdown ⊗ Ps ← Ps)
        blocks(cd)[I((1, -1, Q-P))] .= 1
        blocks(cd)[I((0, 0, 2*Q-P))] .= 1
    
        @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
        @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
        twosite = twosite_up + twosite_down
    else
        Vs = Vect[I]((1, 1 / 2, Q) => 1)

        c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
        blocks(c⁺)[I((1, 1 // 2, Q-P))] .= 1
        blocks(c⁺)[I((0, 0, 2*Q-P))] .= sqrt(2)

        c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
        blocks(c)[I((1, 1 / 2, Q-P))] .= 1
        blocks(c)[I((0, 0, 2*Q-P))] .= sqrt(2)

        @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    end

    return twosite
end

function OSInteraction(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 0, 2*Q-P))] .= 1
    else
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 0, 2*Q-P))] .= 1
    end

    return onesite
end

function Number(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        n = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(n)[I((0, 0, 2*Q-P))] .= 2
        blocks(n)[I((1, 1, Q-P))] .= 1
        blocks(n)[I((1, -1, Q-P))] .= 1
    else
        n = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(n)[I((0, 0, 2*Q-P))] .= 2
        blocks(n)[I((1, 1 // 2, Q-P))] .= 1
    end

    return n
end

function SymSpace()
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)

    return I, Ps
end

function Hopping()
    I, Ps = SymSpace()
    Vs = Vect[I]((1, 1 / 2) => 1)

    c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
    blocks(c⁺)[I((1, 1 // 2))] = [1.0+0.0im 0.0+0.0im]
    blocks(c⁺)[I((0, 0))] = [0.0+0.0im; sqrt(2)+0.0im;;]

    c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
    blocks(c)[I((1, 1 // 2))] = [1.0+0.0im; 0.0+0.0im;;]
    blocks(c)[I((0, 0))] = [0.0+0.0im sqrt(2)+0.0im]

    @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    
    return twosite
end

function OSInteraction()
    I, Ps = SymSpace()

    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 1.0] 

    return onesite
end

function Number()
    I, Ps = SymSpace()

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0] 
    blocks(n)[I((1, 1 // 2))] .= 1.0

    return n
end


# ONEBAND #

function hamiltonian(simul::OB_Sim)
    t = simul.t
    u = simul.u
    μ = simul.μ
    L = simul.period
    spin = get(simul.kwargs, :spin, false)

    D_hop = length(t)
    D_int = length(u)
    
    if hasproperty(simul, :P)
        P = simul.P
        Q = simul.Q
        if iseven(P)
            T = Q
        else 
            T = 2*Q
        end
        cdc = Hopping(P,Q,spin)    
        n = Number(P,Q,spin)
        OSI = OSInteraction(P,Q,spin)
    else
        T = 1
        cdc = Hopping()
        n = Number()
        OSI = OSInteraction()
    end

    twosite = cdc + cdc'
    onesite = u[1]*OSI - μ*n

    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    
    H = @mpoham sum(onesite{i} for i in vertices(InfiniteChain(T)))
    if L == 0
        for range_hop in 1:D_hop
            h = @mpoham sum(-t[range_hop]*twosite{i,i+range_hop} for i in vertices(InfiniteChain(T)))
            H += h
        end
        for range_int in 2:D_int
            h = @mpoham sum(u[range_int]*nn{i,i+range_int} for i in vertices(InfiniteChain(T)))
            H += h
        end
    elseif D_hop==1 && D_int==1
        h = @mpoham sum(-t[1]*twosite{i,i+1} -t[1]*twosite{i,i+L} for i in vertices(InfiniteChain(T)))
        H += h
    else
        return error("No extended models in 2D.")
    end

    return H
end


function non_abelian_mps(operator, P::Int64, max_dimension::Int64, spin::Bool)
    Ps = operator.pspaces
    L = length(Ps)
    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    if !spin
        Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((0,0,0)=>1)     # find maximal virtual space
        for i in 0:1
            for j in 0:1//2:3//2
                for k in -(L*P):1:(L*P)
                    Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((i,j,k)=>max_dimension) ⊕ Vmax
                end
            end
        end
    else
        Vmax = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((0,0,0)=>1)
        for i in 0:1
            for j in -L:1:L
                for k in -(L*P):1:(L*P)
                    Vmax = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((i,j,k)=>max_dimension) ⊕ Vmax
                end
            end
        end
    end

    V_max = copy(V)

    for i in 1:length(V_right)
        V_max[i] = Vmax
    end

    V_trunc = TensorKit.infimum.(V,V_max)

    return InfiniteMPS(Ps, V_trunc)
end
