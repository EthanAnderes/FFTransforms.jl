


# features of the input output arrays to plan(w::𝕎)
# ====================================

@inline eltype_in(w::𝕎{Tf,d}) where {Tf,d}  = Tf

@inline eltype_out(w::𝕎{Tf,d}) where {Tf,d} = Complex{real(Tf)}

@inline size_in(w::𝕎) = w.sz

size_out(w::𝕎{Tf}) where {Tf<:FFTC} = w.sz

function size_out(w::𝕎{Tf,d})::NTuple{d,Int} where {Tf<:FFTR,d}
    ir = findfirst(w.region)
    return map(w.sz, tuple(1:d...)) do nᵢ, i
        i==ir ? nᵢ÷2+1 : nᵢ
    end
end


# features of the grid from w.period
# =================================

Δpix(w::𝕎)  = @. w.period / w.sz

Δfreq(w::𝕎) = @. 2π / w.period

nyq(w::𝕎)   = π ./ Δpix(w)

# Note: this gives the area element of 
# only the fourier tranformed coordinates
Ωx(w::𝕎)  = prod(Δr[1] for Δr in zip(Δpix(w), w.region) if Δr[2])

Ωk(w::𝕎)  = prod(Δr[1] for Δr in zip(Δfreq(w), w.region) if Δr[2])


# 𝕎 scalings
# =================================

"`nv_scale(w::𝕎)->Number` returns the multiplicative scale of the inverse of w::𝕎"
function inv_scale(w::𝕎{Tf,d}) where {Tf,d}
    ifft_normalization = FFTW.normalization(
                real(Tf), 
                w.sz, 
                tuple(findall(w.region)...)
            )
    return ifft_normalization / w.scale
end

function unitary_scale(w::𝕎{Tf,d}) where {Tf,d}
    return prod(1/√i[1] for i in zip(w.sz, w.region) if i[2])
end

ordinary_scale(w::𝕎{Tf,d}) where {Tf,d} = Ωx(w) / ((2π)^(sum(w.region)/2))

# Test that inv_scale(ordinary_scale(w))== Ωk(w) / ((2π)^(sum(w.region)/2))



# pix grid 
# =================================

# low level 
# -----------

function pix(n::Int, p::Tp)::Vector{Tp} where Tp
    Δx = p/n
    x  = (0:n-1) * Δx |> collect
    return x 
end

function pix(n::NTuple{d,Int}, p::NTuple{d,Tp})::NTuple{d,Vector{Tp}} where {d,Tp}
    return map(pix, n, p)
end

# main API
# -----------

pix(w::𝕎{Tf,d}) where {d,Tf} = pix(w.sz, w.period)


function fullpix(i::Int, w::𝕎{Tf,d,Tsf,Tp})::Array{Tp,d}  where {Tf,d,Tsf,Tp}
    xi     = pix(w, w.p)
    xifull = zeros(Tp, w.sz)
    for Ic ∈ CartesianIndices(xifull)
        xifull[I] = getindex(xi[i], Ic.I[i])
    end
    return xifull
end

function fullpix(w::𝕎{Tf,d,Tsi,Tp})::NTuple{d,Array{Tp,d}} where {Tf,d,Tsi,Tp}
    return map(i->fullpix(i, w), tuple(1:d...))
end


# frequency grid 
# =================================

# low level 
# -----------

function freq(n::Int, p::Tp)::Vector{Tp} where Tp
    Δx   = p/n
    Δk   = 2π/p
    nyq  = 2π/(2Δx)
    k    = _fft_output_index_2_freq.(1:n, n, p)
    return k
end

function freq(n::NTuple{d,Int}, p::NTuple{d,Tp}, region::NTuple{d,Bool}) where {d,Tp}
    k = map(n, p, region) do nᵢ, pᵢ, rᵢ
        rᵢ ? freq(nᵢ, pᵢ) : pix(nᵢ, pᵢ) 
    end 
    return k::NTuple{d,Vector{Tp}}
end


function rfreq(n::Int, p::Tp)::Vector{Tp} where Tp
    freq(n, p)[1:(n÷2+1)]
end


function rfreq(n::NTuple{d,Int}, p::NTuple{d,Tp}, region::NTuple{d,Bool}) where {d,Tp}
    ir = findfirst(region)
    k  =  map(n, p, region, tuple(1:d...)) do nᵢ, pᵢ, rᵢ, i
        i==ir ? rfreq(nᵢ, pᵢ) : 
        rᵢ    ? freq(nᵢ, pᵢ)  : pix(nᵢ, pᵢ) 
    end
    return k::NTuple{d,Vector{Tp}}
end


# main API
# -----------

freq(w::𝕎{Tf,d}) where {d, Tf<:FFTR} = rfreq(w.sz, w.p, w.region)

freq(w::𝕎{Tf,d}) where {d, Tf<:FFTC} = freq(w.sz, w.p, w.region)

function fullfreq(i::Int, w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    ki  = freq(w)[i]
    kifull = zeros(Tp, size_out(w))
    for Ic ∈ CartesianIndices(kifull)
        kifull[I] = getindex(ki, Ic.I[i])
    end
    return kifull
end

function fullfreq(w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    map(i->fullfreq(i,w,p), tuple(1:d...))::NTuple{d,Array{Tp,d}}
end

function wavenum(w::𝕎{Tf,d,Tsf,Tp}) where {Tf,d,Tsf,Tp}
    k  = freq(w)
    λ  = zeros(Tp, size_out(w))
    for Ic ∈ CartesianIndices(λ)
        λ[I] = sqrt(sum(abs2, getindex.(k,Ic.I)))
    end
    return λ
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


