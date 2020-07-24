# FFTransforms

Under construction ...

This package defines a type `𝕎{T, d, S, P}` which corresponds to a struct that holds enough information to completely specify a fast Fourier transform of an `Array{T, d}`. The type parameters `{T,d}` correspond to the input Array type parameters. The type parameters `S<:Number` and `P<:Real` correspond, respectively, to the storage type of the fft normalizing constant and the period of pixel coordinate region. Instances of `𝕎` are intended as lightweight objects that can be used for easily generating fast Fourier transform plans of arrays, or can be used with XFields. 

`𝕎{T, d, S, P}` is a subtype of `Transform{T,d}`, from `XFields.jl`, and have the following methods predefined (the necessary interface methods for concrete subtypes of `Transform{T,d}`): 
* `size_in(W::𝕎)` returns the size of the fft input.
* `size_out(W::𝕎)` returns the size of the fft output.
* `eltype_in(W::𝕎)` returns the eltype of the fft input.
* `eltype_out(W::𝕎)`  returns the eltype of the fft output.
* `plan(W::𝕎)`.

The method `plan(W::𝕎)` returns a `pW::FFTplan` which does the actual fft forward/backward transforms on arrays via `*` abd `\`.

Besides providing a convenient constructor for fft plans this package also pre-defines some grid geometric information for each `W::𝕎`.  

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
plan(F)*f ≈ ∫∫ exp(-√(-1)(k₂⋅x₂+k₃⋅x₃))f(x₁,x₂,x₃,x₄)dx₂dx₃/(2π)
```

where the region of integration in the above integral is `[0,π]×[0,2π]`. In this case the function `Ωx(F) == (π/128) * (2π/16)` which approximates `dx₂dx₃` in the Riemann sum of the above integral.

```julia 
pF  = plan(F)
f   = randn(Float64, 16, 128, 16, 4)
g   = pF * f
f′  = pF \ g
```



(Note this package defines 𝕎 which is `\BbbW<tab-complete>` in sublime but is `\bbW<tab-complete>`in the julia REPL)

