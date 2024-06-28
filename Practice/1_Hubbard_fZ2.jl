using TensorKit, MPSKit, MPSKitModels

d = 2 # Physical dimension
L = 16 # Length spin chain
D = 12 # Bond dimension

H = transverse_field_ising()

algorithm = DMRG2(); # Summon DMRG
Ψ = FiniteMPS(L, ℂ^d, ℂ^D) # Random MPS ansatz with bond dimension D
Ψ₀,_ = find_groundstate(Ψ, H, algorithm);

FiniteChain(length::Integer=1)

hubbard_model([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}],
              [spin_symmetry::Type{<:Sector}], [lattice::AbstractLattice];
              t, U, mu, n)