using LinearAlgebra
using QuantumLattices
using ExactDiagonalization


unitcell = QuantumLattices.Lattice([0.0, 0.0]; vectors=[[1.0, 0.0]])
lattice = QuantumLattices.Lattice(unitcell, (12, ), ('o',))
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))
quantumnumber = ParticleNumber(length(lattice))
t = Hopping(:t, -1.0, 1)
U = Hubbard(:U, 8.0)
@time ed = ED(lattice, hilbert, (t, U), quantumnumber)
@time eigensystem = eigen(ed; nev=1)
print(eigensystem.values[1])

#L=6, U=0, [o, p]
#-6.987918414869875
#-7.999999999999998

#L=6, U=4 [o, p]
#-3.0925653195053977
#-3.668706178872949

#L=12 U=8 [o]
#=
9.354387 seconds (77.57 M allocations: 17.085 GiB, 6.27% gc time)
26.913246 seconds (2.57 k allocations: 1.610 GiB, 0.13% gc time)
-3.7283960387196835
=#