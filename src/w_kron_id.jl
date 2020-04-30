

# Adds a lightweight layer between a container for concrete plan and the inputs 
# to the planning methods. 
# This allows one to easily store 𝕎 as an field in an array type 
# wrapper. 

# plans via kron's of FFTs and Identity operators
# ================================================

# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) untill 
# passed to the plan method

# 𝕎 holds sz (size of the input array) 
# and region (which determines which axes get FFT'd)
# --------------------------
struct 𝕎{Tf<:FFTN, d, Tsf<:Number, Tp<:Real}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool}
	scale::Tsf 
	period::NTuple{d,Tp}

	function 𝕎{Tf,d}(sz::NTuple{d,Int}, rg::NTuple{d,Bool}, sc::Tsf, pd::NTuple{d,Tp}) where {Tf<:FFTN,d,Tsf,Tp}
		new{Tf,d,Tsf,Tp}(sz,rg,sc,pd)
	end
end 


# Construct directly from 𝕎{Tf}(sz,rg,sc,pd) or 
# alternatively with a kron of scale * 1-d 𝕎 


𝕎(::Type{Tf}, sz::Int)          where Tf<:FFTN = 𝕎{Tf,1}((sz,), (true,), true, (sz,))
𝕎(::Type{Tf}, sz::Int, p::Real) where Tf<:FFTN = 𝕎{Tf,1}((sz,), (true,), true, (p,))

𝕎(sz::Int) = 𝕎(C64, sz)
𝕎(sz::Int, p::Real) = 𝕎(C64, sz, p)

𝕎32(sz::Int) = 𝕎(C32, sz)
𝕎32(sz::Int, p::Real) = 𝕎(C32, sz, p)

r𝕎(sz::Int) = 𝕎(F64, sz)
r𝕎(sz::Int, p::Real) = 𝕎(F64, sz, p)

r𝕎32(sz::Int) = 𝕎(F32, sz)
r𝕎32(sz::Int, p::Real) = 𝕎(F32, sz, p)


# 𝕀 only encode sz and period of the grid
# --------------------------
struct 𝕀{d,Tp<:Real}
	sz::NTuple{d,Int}
	period::NTuple{d,Tp} 
	𝕀{d}(sz::NTuple{d,Int},period::NTuple{d,Tp}) where {d,Tp} = new{d,Tp}(sz,period)
end

𝕀(sz::Int) = 𝕀{1}((sz,),(sz,))
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

function Base.:*(s::Number, w::𝕎{Tf,d}) where {d,Tf}
	return 𝕎{Tf,d}(w.sz, w.region, s*w.scale, w.period)
end

Base.:*(w::𝕎, s::Number) = s*w

function unscale𝕎(w::𝕎{Tf,d}) where {Tf,d}
	return 𝕎{Tf,d}(w.sz, w.region, true, w.period)
end

function real𝕎(w::𝕎{Tf,d}) where {Tf,d}
	return 𝕎{real(Tf),d}(w.sz, w.region, w.scale, w.period)
end

function complex𝕎(w::𝕎{Tf,d}) where {Tf,d}
	return 𝕎{Complex{real(Tf)},d}(w.sz, w.region, w.scale, w.period)
end


# obtaining a planned FFT from 𝕎 
# ===========================================

# fixme: Getting type stability is hard here. 

@inline eltype_in(w::𝕎{Tf,d}) where {Tf,d}  = Tf

@inline eltype_out(w::𝕎{Tf,d}) where {Tf,d} = Complex{real(Tf)}

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTR,Tsf} 
	Ti   = Complex{Tf}
	Tsi  = promote_type(Tf, Tsf) 
	FT   = FFTW.rFFTWPlan{Tf,-1,false,d}
	IT   = FFTW.rFFTWPlan{Ti,1, false,d}
	rtn_type = FFTplan{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTC,Tsf} 
	Ti  = Tf
	Tsi = promote_type(real(Tf), Tsf) 
	FT = FFTW.cFFTWPlan{Tf,-1,false,d}
	IT = FFTW.cFFTWPlan{Ti,1, false,d}
	rtn_type = FFTplan{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 




