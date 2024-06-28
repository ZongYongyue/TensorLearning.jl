using TensorKit
using ITensors
using BenchmarkTools
using MPSKitModels
using MPSKit
d = 100
A = rand(d,d,d); B = rand(d,d,d); C = rand(d,d,d);

function ITE(A, B, C)
    d = 100
    i,j,k,l,m,n = Index(d),Index(d),Index(d),Index(d),Index(d),Index(d)
    IA = ITensor(A, i,j,k); IB = ITensor(B, j,l,m); IC = ITensor(C, k,m,n);
    return IA*IB*IC
end



function TeK(A,B,C)
    @tensor D[i, l, n] := A[i, j, k] * B[j, l, m] * C[k, m, n]
    return D
end

@benchmark ITE(A,B,C)#$IA*$IB*$IC
#@benchmark TeK($A,$B,$C)

