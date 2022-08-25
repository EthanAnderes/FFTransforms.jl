

# randn_in and rand_out
# =====================================

function randn_in(tm::TM) where {Tf<:Real, TM<:𝕎{Tf}}
    wx = randn(eltype_in(tm), size_in(tm)) ./ sqrt.(Ωpix(tm))
    wx
end

# This needs testing 
function randn_out(tm::TM) where {Tf<:Complex, TM<:𝕎{Tf}}
    wk = randn(eltype_in(tm), size_in(tm)) ./ sqrt.(Ωfreq(tm))
    wk
end


# dot_in and dot_out
# =====================================

function dot_in(f::Xfield{FT}, g::Xfield{FT}) where {T<:Real, FT<:𝕎{T}}
    tm     = fieldtransform(f)
    fdata  = fielddata(MapField(f))
    gdata  = fielddata(MapField(g))
    Ω      = Ωpix(tm)
    return  sum_kbn(fdata .* gdata .* Ω)
end


function dot_in(f::Xfield{FT}, g::Xfield{FT}) where {T<:Complex, FT<:𝕎{T}}
    tm     = fieldtransform(f)
    fdata  = fielddata(MapField(f))
    gdata  = fielddata(MapField(g))
    Ω      = Ωpix(tm)
    return  sum_kbn(fdata .* conj.(gdata) .* Ω)
end

# TODO: 
# function dot_out(f::Xfield{FT}, g::Xfield{FT}) where {T<:Real, FT<:𝕎{Complex{T}}}
#     tm     = fieldtransform(f)
#     fdata  = fielddata(FourierField(f))
#     gdata  = fielddata(FourierField(g))
#     Ω      = Ωfreq(tm)
#     return  sum_kbn(fdata .* conj.(gdata) .* Ω)
# end
