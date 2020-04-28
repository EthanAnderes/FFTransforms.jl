

# pix and fourier grid stats 
# -----------------------------

function grid(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    y = map(w.sz, p) do nᵢ, pᵢ
        Δx     = pᵢ/nᵢ
        Δk     = 2π/pᵢ
        nyq    = 2π/(2Δx)
        (Δx=Δx, Δk=Δk, nyq=nyq) 
    end
    Δki     = tuple((yi.Δk for yi ∈ y)...)
    Δxi     = tuple((yi.Δx for yi ∈ y)...)
    nyqi    = tuple((yi.nyq for yi ∈ y)...)
    nxi     = w.sz
    nki     = map(length, freq(w, p))
    return (Δxi=Δxi, Δki=Δki, nyqi=nyqi, nxi=nxi, nki=nki)
end


# pix grid 
# -----------------------------

# low level 

function pix(n::Int, p::T) where T<:Real
    Δx = p/n
    x  = (0:n-1) * Δx |> collect
    return T.(x) 
end

function pix(n::NTuple{d,Int}, p::NTuple{d,T}) where {d,T<:Real}
    return map(pix, n, p)
end

# main API

function pix(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    rTf = real(Tf) 
    return pix(w.sz, rTf.(p))
end


function fullpix(i::Int, w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    xifull = zeros(real(Tf), w.sz)
    xi     = pix(w, p)
    for I ∈ CartesianIndices(xifull)
        xifull[I] = getindex(xi[i], I.I[i])
    end
    return xifull
end

function fullpix(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    map(i->fullpix(i, w, p), tuple(1:d...))::NTuple{d,Array{real(Tf),d}}
end



# frequency grid 
# =================================

# low level 
# -----------

function freq(n::Int, p::T) where T<:Real
    Δx   = p/n
    Δk   = 2π/p
    nyq  = 2π/(2Δx)
    k    = _fft_output_index_2_freq.(1:n, n, p)
    return T.(k)
end

function freq(n::NTuple{d,Int}, p::NTuple{d,T}, region::NTuple{d,Bool}) where {d,T<:Real}
    return map(n, p, region) do ni, pi, ri
        !ri ? pix(ni, pi) : freq(ni, pi)
    end
end

function freq(n::NTuple{d,Int}, p::NTuple{d,T}) where {d,T<:Real}
    freq(n, p, tuple(trues(d)...))
end

function rfreq(n::NTuple{d,Int}, p::NTuple{d,T}, region::NTuple{d,Bool}) where {d,T<:Real}
    ir = findfirst(region)
    return map(n, p, region, tuple(1:d...)) do ni, pi, ri, i
        !ri   ? pix(ni, pi) : 
        i==ir ? freq(ni, pi)[1:(ni÷2+1)] : freq(ni, pi)
    end::NTuple{d,Array{T,1}}
end

# main API
# -----------

function freq(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d, Tf<:FFTWReal, T} 
    return rfreq(w.sz, Tf.(p), w.region)
end

function freq(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d, Tf<:FFTWComplex, T} 
    rTf = real(Tf) 
    return freq(w.sz, rTf.(p), w.region)
end

function fullfreq(i::Int, w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    ki  = freq(w,p)
    nki = map(length,ki) 
    kifull = zeros(real(Tf), nki)
    for I ∈ CartesianIndices(kifull)
        kifull[I] = getindex(ki[i], I.I[i])
    end
    return kifull
end

function fullfreq(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    map(i->fullfreq(i,w,p), tuple(1:d...))::NTuple{d,Array{real(Tf),d}}
end

function wavenum(w::𝕎{Tf,d}, p::NTuple{d,T}) where {d,Tf,T}
    ki  = freq(w,p)
    nki = map(length, ki)
    λ  = zeros(T, nki)
    for I ∈ CartesianIndices(λ)
        λ[I] = sqrt(sum(abs2, getindex.(ki,I.I)))
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


