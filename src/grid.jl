
# TODO: add impulse response iterators

# features of the input output arrays to plan(w::𝕎)
# ====================================


# features of the grid from w.period
# =================================

function Δpix(w::𝕎{Tf}) where {Tf}
    rTf = real(Tf) 
    rTf.(w.period ./ w.sz)
end

function Δfreq(w::𝕎{Tf}) where {Tf} 
    rTf = real(Tf) 
    rTf.((2π) ./ w.period)
end

function nyq(w::𝕎{Tf}) where {Tf} 
    rTf = real(Tf) 
    rTf.(π ./ Δpix(w))
end

# Note: this gives the area element of 
# only the fourier tranformed coordinates
Ωpix(w::𝕎) = prod(Δr[1] for Δr in zip(Δpix(w), w.region) if Δr[2])

Ωfreq(w::𝕎) = prod(Δr[1] for Δr in zip(Δfreq(w), w.region) if Δr[2])


# 𝕎 scalings
# =================================

function unscale(w::𝕎{Tf,d}) where {Tf,d}
    𝕎{Tf,d}(w.sz, w.region, true, w.period)
end


function Base.real(w::𝕎{Tf,d}) where {Tf,d}
    𝕎{real(Tf),d}(w.sz, w.region, w.scale, w.period)
end

function Base.complex(w::𝕎{Tf,d}) where {Tf,d}
    𝕎{Complex{real(Tf)},d}(w.sz, w.region, w.scale, w.period)
end

"`nv_scale(w::𝕎)->Number` returns the multiplicative scale of the inverse of w::𝕎"
function inv_scale(w::𝕎{Tf}) where {Tf}
    rTf = real(Tf)
    ifft_normalization = FFTW.normalization(
                rTf, 
                w.sz, 
                tuple(findall(w.region)...)
            )
    return rTf(ifft_normalization / w.scale)
end

function unitary_scale(w::𝕎{Tf}) where {Tf}
    rTf = real(Tf)
    return rTf(prod(1/√i[1] for i in zip(w.sz, w.region) if i[2]))
end

function ordinary_scale(w::𝕎{Tf}) where {Tf} 
    rTf = real(Tf)
    rTf(Ωpix(w) / ((2π)^(sum(w.region)/2)))
end

# Test that inv_scale(ordinary_scale(w))== Ωfreq(w) / ((2π)^(sum(w.region)/2))




# pix grid 
# =================================

# low level 
# -----------

function pix(n::Int, p::Real)
    Δx = p/n
    x  = (0:n-1) * Δx |> collect
    return x 
end


# main API
# -----------

function pix(w::𝕎{Tf,d}) where {d,Tf}
    rTf = real(Tf)
    return map(w.sz, w.period) do nᵢ, pᵢ
        rTf.(pix(nᵢ, pᵢ))
    end::NTuple{d,Array{rTf,1}}
end

function fullpix(i::Int, w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    xi     = pix(w)[i]
    rTf    = real(Tf)
    xifull = zeros(rTf, w.sz)
    for Ic ∈ CartesianIndices(xifull)
        xifull[Ic] = getindex(xi, Tuple(Ic)[i])
    end
    return xifull::Array{rTf,d}
end

function fullpix(w::𝕎{Tf,d,Tsi,Tp}) where {Tf,d,Tsi,Tp}
    rTf = real(Tf)
    return map(tuple(1:d...)) do i 
        fullpix(i, w) 
    end::NTuple{d,Array{rTf,d}}
end


# frequency grid 
# =================================

# low level 
# -----------

function freq(n::Int, p::Real)
    k    = _fft_output_index_2_freq.(1:n, n, p)
    return k
end

function rfreq(n::Int, p::Tp) where Tp
    freq(n, p)[1:(n÷2+1)]
end


# main API
# -----------

function freq(w::𝕎{Tf,d}) where {d, Tf<:FFTC}
    rTf = real(Tf)
    return map(w.sz, w.period, w.region) do nᵢ, pᵢ, rᵢ
        kᵢ = rᵢ ? freq(nᵢ, pᵢ) : pix(nᵢ, pᵢ) 
        rTf.(kᵢ)
    end::NTuple{d,Array{rTf,1}}
end

function freq(w::𝕎{Tf,d}) where {d, Tf<:FFTR}
    ir = findfirst(w.region)
    return map(w.sz, w.period, w.region, tuple(1:d...)) do nᵢ, pᵢ, rᵢ, i
        kᵢ = (i==ir) ? rfreq(nᵢ, pᵢ) : 
                  rᵢ ?  freq(nᵢ, pᵢ) : pix(nᵢ, pᵢ)
        Tf.(kᵢ)
    end::NTuple{d,Array{Tf,1}}
end

function fullfreq(i::Int, w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    ki  = freq(w)[i]
    rTf = real(Tf)
    kifull = zeros(rTf, size_out(w))
    for Ic ∈ CartesianIndices(kifull)
        kifull[Ic] = getindex(ki, Tuple(Ic)[i])
    end
    return kifull::Array{rTf,d}
end

function fullfreq(w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    return map(tuple(1:d...)) do i
        fullfreq(i,w)
    end::NTuple{d,Array{real(Tf),d}}
end

function wavenum(w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    k   = freq(w)
    rTf = real(Tf)
    λ   = zeros(rTf, size_out(w))
    for Ic ∈ CartesianIndices(λ)
        λ[Ic] = sqrt(sum(abs2, getindex.(k,Tuple(Ic))))
    end
    return λ::Array{rTf,d}
end



# Internal function 
# -----------------------------------
#`k_pre = _fft_output_index_2_freq.(1:n, n, p)` computes the 1-d freq
function _fft_output_index_2_freq(ind, nside, period)
    kpre = (2π / period) * (ind - 1)
    nyq  = (2π / period) * (nside/2)

    # • Both of the following options are valid
    # • Both options return the same value when nside is odd
    # • Using (kpre <= nyq) sets the convention  that the 
    #   frequencies fall in (-nyq, nyq] when nside is even
    # • Using (kpre < nyq) sets the convention  that the 
    #   frequencies fall in [-nyq, nyq) when nside is even
    #----------------
    # • Here are the two options: 
    return  ifelse(kpre <= nyq, kpre, kpre - 2nyq) # option 1
    # return ifelse(kpre < nyq, kpre, kpre - 2nyq)  # option 2
end

# function _get_npd(;nᵢ, pᵢ=nothing, Δxᵢ=nothing)
#     @assert !(isnothing(pᵢ) & isnothing(Δxᵢ)) "either pᵢ or Δxᵢ needs to be specified (note: pᵢ = Δxᵢ .* nᵢ)"
#     d = length(nᵢ)
#     if isnothing(pᵢ)
#         @assert d == length(Δxᵢ) "Δxᵢ and nᵢ need to be tuples of the same length"
#         pᵢ = tuple((prod(xn) for xn in zip(Δxᵢ,nᵢ))...)
#     end
#     @assert d == length(pᵢ) "pᵢ and nᵢ need to be tuples of the same length"
#     nᵢ, pᵢ, d
# end



# #%% Used for constructing the covariance matrix of a subset of frequencies
# function get_rFFTimpulses(::Type{F}) where {T,nᵢ,pᵢ,dnᵢ,F<:rFFTgeneric{T,nᵢ,pᵢ,dnᵢ}}
#     g  = Grid(F)
#     CI = CartesianIndices(Base.OneTo.(g.nki))
#     LI = LinearIndices(Base.OneTo.(g.nki))

#     function _get_dual_k(k,n) 
#         dk = n-k+2
#         mod1(dk,n)
#     end 

#     function get_dual_ci(ci::CartesianIndex{dnᵢ}) 
#         return CartesianIndex(map(_get_dual_k, ci.I, g.nxi))
#     end 

#     function rFFTimpulses(ci::CartesianIndex{dnᵢ})
#         rimpls = zeros(Complex{T}, g.nki...)
#         cimpls = zeros(Complex{T}, g.nki...)
#         dual_ci = get_dual_ci(ci)
#         if (ci==first(CI)) || (ci==dual_ci)
#             rimpls[ci]  = 1
#         elseif dual_ci ∈ CI
#             rimpls[ci]  = 1/2
#             cimpls[ci]  = im/2
#             rimpls[dual_ci]  =  1/2
#             cimpls[dual_ci]  = -im/2
#         else
#             rimpls[ci]  = 1/2
#             cimpls[ci]  = im/2
#         end
#         return rimpls, cimpls
#     end

#     return rFFTimpulses, CI, LI, get_dual_ci
# end

