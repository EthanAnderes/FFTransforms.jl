
# Adds a lightweight layer between a container for concrete plan and the inputs 
# to the planning methods. 
# This allows one to easily store 𝕎 as an field in an array type 
# wrapper. 

# plans via kron's of FFTs and Identity operators
# ================================================

# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) until a final
# scalar multplier on the right triggers plan creation

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


𝕎(::Type{Tf}, n::Int)            where Tf<:FFTN = 𝕎{T,1}((n,), (true,), true, (n,))
𝕎(::Type{Tf}, n::Int, p::Number) where Tf<:FFTN = 𝕎{T,1}((n,), (true,), true, (p,))

𝕎(n::Int) = 𝕎(C64, n)
𝕎(n::Int, p::Number) = 𝕎(C64, n, p)

𝕎32(n::Int) = 𝕎(C32, n)
𝕎32(n::Int, p::Number) = 𝕎(C32, n, p)

r𝕎(n::Int) = 𝕎(F64, n)
r𝕎(n::Int, p::Number) = 𝕎(F64, n, p)

r𝕎32(n::Int) = 𝕎(F32, n)
r𝕎32(n::Int, p::Number) = 𝕎(F32, n, p)


# 𝕀 only encode sz (size of the input array)
# --------------------------
struct 𝕀{d}
	sz::NTuple{d,Int} 
end

𝕀(n::Vararg{Int,d}) where {d} = 𝕀{d}(n)


# Define the lazy kron operators. The last mult on the right by a scalar 
# is the trigger for generating a concrete plan
# --------------------------

function ⊗(i::𝕀{n}, j::𝕀{d}) where {n,d} 
	sz  = tuple(i.sz..., j.sz...)
	return 𝕀{d+n}(sz)
end

function ⊗(i::𝕀{n}, w::𝕎{Tf,d,Tsf,Tp}) where {n,Tf,d,Tsf,Tp} 
	sz     = tuple(i.sz..., w.sz...)
	region = tuple(falses(n)..., w.region...)
	period = tuple(Tp.(n)..., w.period...)
	return 𝕎{Tf,d+n,Tsf,Tp}(sz, region, w.scale, period)
end

function ⊗(w::𝕎{Tf,d,Tsf,Tp}, i::𝕀{n}) where {n,Tf,d,Tsf,Tp}
	sz     = tuple(w.sz..., i.sz...)
	region = tuple(w.region..., falses(n)...)
	period = tuple(w.period..., Tp.(n)...)
	return 𝕎{Tf,d+n,Tsf,Tp}(sz, region, w.scale, period)
end

# The element type of the first 𝕎 (reading left to right) determines 
# the overall type of the transform 
# Do we want this to promote on the real type? 
# .... so 𝕎{R<:FFTR,d}    ⊗ 𝕎{T,d} -> promote_type(R,real(T))
# ....and 𝕎{R<:FFTC,d} ⊗ 𝕎{T,d} -> Complex{promote_type(real(R),real(T))}

function ⊗(w::𝕎{Tf,d,Tsf,Tp}, v::𝕎{Rf,n,Rsf,Rp}) where {Tf<:FFTN,Rf<:FFTN,d,n,Tsf,Rsf,Tp,Rp} 
	sz     = tuple(w.sz..., v.sz...)
	region = tuple(w.region..., v.region...)
	Tsf′   = promote_type(Tsf, Rsf)
	Tp′    = promote_type(Tp, Rp)
	scale  = w.scale * v.scale 
	period = tuple(Tp′.(w.period)..., Tp′.(v.period)...)
	return 𝕎{Tf,d+n, Tsf′, Tp′}(sz, region, scale, period)
end

# function ⊗(w::𝕎{R,d}, v::𝕎{T,n}) where {d,n,R<:FFTN,T<:FFTN} 
# 	sz     = tuple(w.sz..., v.sz...)
# 	region = tuple(w.region..., v.region...)
# 	scale  = w.scale * v.scale 
# 	period = tuple(w.period..., v.period...)
# 	return 𝕎{R,d+n}(sz, region, scale, period)
# end


function Base.:*(s::Number, w::𝕎{Tf,d}) where {d,Tf}
	return 𝕎{Tf,d}(w.sz, w.region, s*w.scale, w.period)
end

Base.:*(w::𝕎, s::Number) = s*w

function unscale𝕎(w::𝕎{T,d}) where {T,d}
	return 𝕎{T,d}(w.sz, w.region, true, w.period)
end

function real𝕎(w::𝕎{T,d}) where {T,d}
	return 𝕎{real(T),d}(w.sz, w.region, w.scale, w.period)
end

function complex𝕎(w::𝕎{T,d}) where {T,d}
	return 𝕎{Complex{real(T)},d}(w.sz, w.region, w.scale, w.period)
end



# ===================


# obtaining a planned FFT from 𝕎 
# ===========================================

# fixme: Getting type stability is hard here. 

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTR,Tsf} 
	Ti   = Complex{Tf}
	Tsi  = promote_type(Tf, Tsf) 
	FT   = FFTW.rFFTWPlan{Tf,-1,false,d}
	IT   = FFTW.rFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTC,Tsf} 
	Ti  = Tf
	Tsi = promote_type(real(Tf), Tsf) 
	FT = FFTW.cFFTWPlan{Tf,-1,false,d}
	IT = FFTW.cFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 




