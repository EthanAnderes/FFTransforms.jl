# FFTransforms

Under construction ...


(Note this package defines 𝕎 which is `\BbbW<tab-complete>` in sublime but is `\bbW<tab-complete>`in the julia REPL)


## FFT `Transform{T,d}` type `𝕎{T, d, S, P}` for XFields

Instances of `𝕎` hold enough information to completely specify a fast Fourier transform of an `Array{T, d}`. They are intended as lightweight objects that can be used as a stand alone lightweight object for easily generating fast Fourier transform plans of arrays, or can be used with XFields. 

### Quick start example 1

```julia
using FFTransforms
n = 128
W = 𝕎(128)     
U = (1/√n) * W # equivalent to unitary_scale(W)*W
```

`W` represents the classic 1-d FFT operating on complex vectors of length `n=128`. `U` represents the unitary version of `W`. `W` and `U` don't operate on vectors themselves. One can generate operators with the function `plan`.

```julia
pW  = plan(W)
pU  = plan(U)
f   = randn(Complex{Float64}, n)
g   = pW * f
h   = pU * f
f′  = pW \ g
f′′ = pU \ h
```

```julia
using LinearAlgebra
norm(f - f′)
norm(f - f′′)
dot(f, f) - dot(h, h)
```

From `W` and `U` one can also retrieve pixel and frequency information on the coordinates of the input and output of the corresponding Fourier operators. 

```julia
pix(W)
freq(W)
nyq(W)
```

```julia
pix(U)
freq(U)
nyq(U)
```

### Quick start example 2

```julia
using FFTransforms
W = 𝕀(16,1.0) ⊗ r𝕎(128,π) ⊗ 𝕎(16,2π) ⊗ 𝕀(4)  
F = ordinary_scale(W) * W 
```

`F` operates on 4-dimensional real arrays `f` as the matrix operation that approximates

```julia
plan(F)*f ≈ (x₁,k₂,k₃,x₄) -> ∫∫ exp(-√(-1)(k₂⋅x₂+k₃⋅x₃))f(x₁,x₂,x₃,x₄)dx₂dx₃/(2π)
```

where the region of integration in the above integral is `[0,π]×[0,2π]`. In this case the function `Ωx(F) == (π/128) * (2π/16)` which approximates `dx₂dx₃` in the Riemann sum of the above integral.

```julia 
pF  = plan(F)
f   = randn(Float64, 16, 128, 16, 4)
g   = pF * f
f′  = pF \ g
```




# Required methods to hook into XFields ...


```
struct NewTransform{Tf,d,...} <: Transform{Tf,d}
    <any fields here necessary for determining the transform>
end
```

* `size_in(nT::NewTransform) -> <size of the storage for the corresponding MapField>`
* `size_out(nT::NewTransform) -> <size of the storage for the corresponding FourierField>`  
* `eltype_in(nT::NewTransform) -> <eltype of the storage field of the corresponding Field>`
* `eltype_out(nT::NewTransform) -> <eltype of the storage for the corresponding FourierField>`
* `plan(nT::NewTransform) * <storage for the corresponding MapField>`
* `plan(nT::NewTransform) \ <storage for the corresponding FourierField>`


Note: if the transform requires custom methods to convert Map <-> Fourier then one can simply define `plan(nT::NewTransform) = nT` and follow up with overloading `*(nT,<storage>)` and `*(nT,<storage>)` for  `nT::NewTransform`.

