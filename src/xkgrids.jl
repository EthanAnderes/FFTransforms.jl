

function pix(n::Int, p::T) where T<:Real
    Δx = p/n
    x  = (0:n-1) * Δx
    return x 
end

function pix(n::NTuple{d,Int}, p::NTuple{d,T}) where {d,T<:Real}
    return map(pix, n, p)
end

function freq(n::Int, p::T) where T<:Real
    Δx   = p/n
    Δk   = 2π/p
    nyq  = 2π/(2Δx)
    k    = _fft_output_index_2_freq.(1:n, n, p)
    return k
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
    rtn = map(n, p, region, 1:d) do ni, pi, ri, i
        !ri   ? pix(ni, pi) : 
        i==ir ? freq(ni, pi)[1:(ni÷2+1)] : freq(ni, pi)
    end
    tuple(rtn...)
end



# update these to accomidate region
# perhaps just make them 1-d


# struct Grid{T,nᵢ,pᵢ,d}
#     Δxi::NTuple{d,T}
#     Δki::NTuple{d,T}
#     xi::NTuple{d,Vector{T}}
#     ki::NTuple{d,Vector{T}}
#     nyqi::NTuple{d,T}
#     Ωx::T
#     Ωk::T
#     # the following are redundant but convenient for quick access to nᵢ,pᵢ,d
#     nki::NTuple{d,Int} # == tuple of densions for the rFFT
#     nxi::NTuple{d,Int} # == nᵢ
#     periodi::NTuple{d,T} # == pᵢ
#     d::Int # == d
# end



#%% specify the corresponding grid geometry
# @generated function Grid(::Type{<:rFFTgeneric{T,nᵢ,pᵢ,d}}) where {T,nᵢ,pᵢ,d}
# function Grid(::Type{T},nᵢ,pᵢ,d) where {T}
#     y = map(nᵢ, pᵢ, 1:d) do n, p, i
#         Δx     = p/n
#         Δk     = 2π/p
#         nyq    = 2π/(2Δx)
#         x      = (0:n-1) * Δx
#         k_pre = _fft_output_index_2_freq.(1:n, n, p)
#         k      = (i == 1) ? k_pre[1:(n÷2+1)] : k_pre
#         (Δx=Δx, Δk=Δk, nyq=nyq, x=x, k=k) 
#     end
#     Δki     = tuple((yi.Δk for yi ∈ y)...)
#     Δxi     = tuple((yi.Δx for yi ∈ y)...)
#     nyqi    = tuple((yi.nyq for yi ∈ y)...)
#     xi      = tuple((yi.x for yi ∈ y)...)
#     ki      = tuple((yi.k for yi ∈ y)...) # note: you might need to reverse the order here...
#     Ωk      = prod(Δki)
#     Ωx      = prod(Δxi)
#     nxi     = nᵢ
#     nki     = map(length, ki)
#     return Grid{T,nᵢ,pᵢ,d}(Δxi, Δki, xi, ki, nyqi, Ωx, Ωk, nki, nxi, pᵢ, d)
# end
 
# function wavenum(::Type{T},nᵢ,pᵢ,d) where {T}
#     g = Grid(T,nᵢ,pᵢ,d)
#     λ = zeros(T, g.nki)
#     for I ∈ CartesianIndices(λ)
#         λ[I] = sqrt(sum(abs2, getindex.(g.ki,I.I)))
#     end
#     λ
# end

# function freq(i::Int, ::Type{T},nᵢ,pᵢ,d) where {T}
#     g = Grid(T,nᵢ,pᵢ,d)
#     kifull = zeros(T, g.nki)
#     for I ∈ CartesianIndices(kifull)
#         kifull[I] = getindex(g.ki[i], I.I[i])
#     end
#     kifull
# end

# function freq(::Type{T},nᵢ,pᵢ,d) where {T}
#     g = Grid(T,nᵢ,pᵢ,d) 
#     map(i->freq(i,T,nᵢ,pᵢ,d), tuple(1:d...))::NTuple{d,Array{T,d}}
# end

# function pix(i::Int, ::Type{T},nᵢ,pᵢ,d) where {T}
#     g = Grid(T,nᵢ,pᵢ,d)
#     xifull = zeros(T, g.nxi)
#     for I ∈ CartesianIndices(xifull)
#         xifull[I] = getindex(g.xi[i], I.I[i])
#     end
#     xifull
# end

# function pix(::Type{T},nᵢ,pᵢ,d) where {T}
#     g = Grid(T,nᵢ,pᵢ,d) 
#     map(i->pix(i,T,nᵢ,pᵢ,d), tuple(1:d...))::NTuple{d,Array{T,d}}
# end

#%% util
#%% ============================================================

# """
# ` nᵢ, pᵢ, d = _get_npd(;nᵢ, pᵢ=nothing, Δxᵢ=nothing)` is used primarily to to check dimensions are valid
# """
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


"""
`k_pre = _fft_output_index_2_freq.(1:n, n, p)` computes the 1-d freq
"""
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
#

