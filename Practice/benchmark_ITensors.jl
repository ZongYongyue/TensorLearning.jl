using ITensors
using ITensorMPS


N=6
function hubbard_1d_o(; N::Int, t=1.0, U=0.0)
    opsum = OpSum()
    for b in 1:(N - 1)
      opsum -= t, "Cdagup", b, "Cup", b + 1
      opsum -= t, "Cdagup", b + 1, "Cup", b
      opsum -= t, "Cdagdn", b, "Cdn", b + 1
      opsum -= t, "Cdagdn", b + 1, "Cdn", b
    end
    if U ≠ 0
      for n in 1:N
        opsum += U, "Nupdn", n
      end
    end
    return opsum
end
function hubbard_1d_p(; N::Int, t=1.0, U=0.0)
    opsum = OpSum()
    for b in 1:(N - 1)
      opsum -= t, "Cdagup", b, "Cup", b + 1
      opsum -= t, "Cdagup", b + 1, "Cup", b
      opsum -= t, "Cdagdn", b, "Cdn", b + 1
      opsum -= t, "Cdagdn", b + 1, "Cdn", b
    end
    opsum -= t, "Cdagup", 1, "Cup", N
    opsum -= t, "Cdagup", N, "Cup", 1
    opsum -= t, "Cdagdn", 1, "Cdn", N
    opsum -= t, "Cdagdn", N, "Cdn", 1
    if U ≠ 0
      for n in 1:N
        opsum += U, "Nupdn", n
      end
    end
    return opsum
end
os = hubbard_1d_o(; N, t=1.0, U=4.0)
sites = siteinds("Electron", N)
H = MPO(os, sites)
psi0 = random_mps(sites; linkdims=20)
nsweeps = 10
maxdim = [10,20,100,100,200, 200, 200, 200, 200, 200,200]
cutoff = [1E-10,1E-10,1E-10,1E-10,1E-10,1E-10,1E-10,1E-10,1E-10]
energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
energy

#L=6, U=0, [o, p]
#-6.987918414739204
#-8.000000000000004

#L=6, U=4, [o, p]
#-4.422071147530258
#-4.698355052221766