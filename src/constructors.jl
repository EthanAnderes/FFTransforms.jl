

# plans via kron's of FFTs and Identity operators
# ================================================

# 2 construction options
#	• Construct directly from 𝕎{Tf}(sz,rg,sc,pd)  
# 	• via kron of 𝕀 and 𝕎 

# ## 𝕎

𝕎(::Type{Tf}, sz::Int)          where Tf<:FFTN = 𝕎{Tf,1}((sz,), (true,), true, (sz,))
𝕎(::Type{Tf}, sz::Int, p::Real) where Tf<:FFTN = 𝕎{Tf,1}((sz,), (true,), true, (p,))
𝕎(sz::Int)          = 𝕎(C64, sz)
𝕎(sz::Int, p::Real) = 𝕎(C64, sz, p)

function 𝕎(::Type{Tf}, sz::NTuple{d,Int}) where {Tf<:FFTN, d} 
	𝕎{Tf,d}(sz, tuple(trues(d)...), true, sz)
end

function 𝕎(::Type{Tf}, sz::NTuple{d,Int}, p::NTuple{d,Tp}) where {Tf<:FFTN, d, Tp<:Real}
	𝕎{Tf,d}(sz, tuple(trues(d)...), true, p)
end


# ## 𝕌 (note: this constructs the correct scaling of 𝕎)

𝕌(::Type{Tf}, sz)    where Tf<:FFTN = (w = 𝕎(Tf, sz);    w*unitary_scale(w))
𝕌(::Type{Tf}, sz, p) where Tf<:FFTN = (w = 𝕎(Tf, sz, p); w*unitary_scale(w))
𝕌(sz)                = (w = 𝕎(sz);    w*unitary_scale(w))
𝕌(sz, p)             = (w = 𝕎(sz, p); w*unitary_scale(w))

# ## 𝔽 (note: this constructs the correct scaling of 𝕎)

𝔽(::Type{Tf}, sz)    where Tf<:FFTN = (w=𝕎(Tf, sz);    w*ordinary_scale(w))
𝔽(::Type{Tf}, sz, p) where Tf<:FFTN = (w=𝕎(Tf, sz, p); w*ordinary_scale(w))
𝔽(sz)                = (w=𝕎(sz);    w*ordinary_scale(w))
𝔽(sz, p)             = (w=𝕎(sz, p); w*ordinary_scale(w))



# 𝕀 only encode sz and period of the grid
# --------------------------
struct 𝕀{d,Tp<:Real}
	sz::NTuple{d,Int}
	period::NTuple{d,Tp} 
	𝕀{d}(sz::NTuple{d,Int},period::NTuple{d,Tp}) where {d,Tp} = new{d,Tp}(sz,period)
end

𝕀(sz::Int)         = 𝕀{1}((sz,),(sz,))
𝕀(sz::Int,p::Real) = 𝕀{1}((sz,), (p,))



# Define the lazy kron operators. The last mult on the right by a scalar 
# is the trigger for generating a concrete plan
# --------------------------

function ⊗(i::𝕀{n,Tp}, j::𝕀{d,Rp}) where {n,d,Tp,Rp} 
	sz     = tuple(i.sz..., j.sz...)
	Tp′    = promote_type(Tp, Rp)
	period = tuple(Tp′.(i.period)..., Tp′.(j.period)...)
	return 𝕀{d+n}(sz,period)
end

function ⊗(i::𝕀{n,Rp}, w::𝕎{Tf,d,Tsf,Tp}) where {n,Tf,d,Tsf,Tp,Rp} 
	sz     = tuple(i.sz..., w.sz...)
	region = tuple(falses(n)..., w.region...)
	Tp′    = promote_type(Tp, Rp)
	period = tuple(Tp′.(i.period)..., Tp′.(w.period)...)
	return 𝕎{Tf,d+n}(sz, region, w.scale, period)
end

function ⊗(w::𝕎{Tf,d,Tsf,Tp}, i::𝕀{n,Rp}) where {n,Tf,d,Tsf,Tp,Rp}
	sz     = tuple(w.sz..., i.sz...)
	region = tuple(w.region..., falses(n)...)
	Tp′    = promote_type(Tp, Rp)
	period = tuple(Tp′.(w.period)..., Tp′.(i.period)...)
	return 𝕎{Tf,d+n}(sz, region, w.scale, period)
end

# The element type of the first 𝕎 (reading left to right) determines 
# the overall type of the transform 
# Do we want this to promote on the real type? 
# .... so 𝕎{R<:FFTR,d} ⊗ 𝕎{T,d} -> promote_type(R,real(T))
# ....and 𝕎{R<:FFTC,d} ⊗ 𝕎{T,d} -> Complex{promote_type(real(R),real(T))}

function ⊗(w::𝕎{Tf,d,Tsf,Tp}, v::𝕎{Rf,n,Rsf,Rp}) where {Tf<:FFTN,Rf<:FFTN,d,n,Tsf,Rsf,Tp,Rp} 
	sz     = tuple(w.sz..., v.sz...)
	region = tuple(w.region..., v.region...)
	scale  = w.scale * v.scale 
	Tp′    = promote_type(Tp, Rp)
	period = tuple(Tp′.(w.period)..., Tp′.(v.period)...)
	return 𝕎{Tf,d+n}(sz, region, scale, period)
end

⊗(w, v, u...) = ⊗(⊗(w, v), u...)

##

function Base.:*(s::Number, w::𝕎{Tf,d}) where {d,Tf}
	𝕎{Tf,d}(w.sz, w.region, s*w.scale, w.period)
end

Base.:*(w::𝕎, s::Number) = s*w

