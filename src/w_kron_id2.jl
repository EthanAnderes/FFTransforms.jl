
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
struct 𝕎{Tf<:FFTWNumber, d, Ts<:Number}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool}
	scale::Ts 
	period::NTuple{d,Tf}

	function 𝕎{Tf,d}(sz::NTuple{d,Int}, rg::NTuple{d,Bool}, sc::Ts, pd::NTuple{d}) where {Tf<:FFTWNumber,d,Ts}
		new{Tf,d,Ts}(sz,rg,sc,T.(pd))
	end
end 


# Construct directly from 𝕎{Tf}(sz,rg,sc,pd) or 
# alternatively with a kron of scale * 1-d 𝕎 

C64 = Complex{Float64}
C32 = Complex{Float32}
F64 = Float64
F32 = Float32

𝕎(::Type{Tf}, n::Int)            where Tf<:FFTWNumber = 𝕎{T,1}((n,), (true,), true, (Tf(n),))
𝕎(::Type{Tf}, n::Int, p::Number) where Tf<:FFTWNumber = 𝕎{T,1}((n,), (true,), true, (Tf(p),))

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

function ⊗(i::𝕀{n}, w::𝕎{Tf,d}) where {n,Tf,d} 
	sz     = tuple(i.sz..., w.sz...)
	region = tuple(falses(n)..., w.region...)
	period = tuple(Tf.(n)..., w.period...)
	return 𝕎{Tf,d+n}(sz, region, w.scale, period)
end

function ⊗(w::𝕎{Tf,d}, i::𝕀{n}) where {n,Tf,d} 
	sz     = tuple(w.sz..., i.sz...)
	region = tuple(w.region..., falses(n)...)
	period = tuple(w.period..., Tf.(n)...)
	return 𝕎{Tf,d+n}(sz, region, w.scale, period)
end

# The element type of the first 𝕎 (reading left to right) determines 
# the overall type of the transform 
# Do we want this to promote on the real type? 
# .... so 𝕎{R<:FFTWReal,d}    ⊗ 𝕎{T,d} -> promote_type(R,real(T))
# ....and 𝕎{R<:FFTWComplex,d} ⊗ 𝕎{T,d} -> Complex{promote_type(real(R),real(T))}
function ⊗(w::𝕎{R,d}, v::𝕎{T,n}) where {d,n,R<:FFTWNumber,T<:FFTWNumber} 
	sz     = tuple(w.sz..., v.sz...)
	region = tuple(w.region..., v.region...)
	scale  = w.scale * v.scale 
	period = tuple(w.period..., v.period...)
	return 𝕎{R,d+n}(sz, region, scale, period)
end

function Base.:*(w::𝕎{Tf,d}, s::S) where {d,Tf,S} = s*w

function Base.:*(s::S, w::𝕎{Tf,d}) where {d,Tf,S}
	return 𝕎{Tf,d}(w.sz, w.region, s, w.period)
end

function unscale(w::𝕎{T,d}) where {T,d}
	return 𝕎{T,d}(w.sz, w.region, true, w.period)
end

function real(w::𝕎{T,d}) where {T,d}
	return 𝕎{real(T),d}(w.sz, w.region, w.scale, w.period)
end

function complex(w::𝕎{T,d}) where {T,d}
	return 𝕎{Complex{real(T)},d}(w.sz, w.region, w.scale, w.period)
end



# ===================

function inverse_scale(w::𝕎{T,d}) where {T,d}
	ifft_normalization = FFTW.normalization(
				real(T), 
				w.sz, 
				tuple(findall(w.region)...)
			)
	return ifft_normalization / w.scale
end

function unitary_scale(w::𝕎{T,d}) where {T,d}
	return prod(1/√i[1] for i in zip(w.sz, w.region) if i[2])
end

rtn_size(w::𝕎{T,d}) where {T<:FFTWComplex,d} = w.sz

function rtn_size(w::𝕎{T,d}) where {T<:FFTWReal,d}
	ir = findfirst(w.region)
    return map(w.sz, w.region, tuple(1:d...)) do ni, ri, i
        i==ir ? ni÷2+1 : ni
    end
end

rtn_type(w::𝕎{T,d}) where {T,d} = Complex{real(T)}


# pix(w::𝕎{T,d}) where {T,d}      
# freq(w::𝕎{T,d}) where {T,d}
# fullpix(w::𝕎{T,d}) where {T,d}      
# fullfreq(w::𝕎{T,d}) where {T,d}
# wavenum(w::𝕎{T,d}) where {T,d}
# Δpix(w::𝕎{T,d}) where {T,d}      
# Δfreq(w::𝕎{T,d}) where {T,d}      


# obtaining a planned FFT from 𝕎 
# ===========================================

# fixme: Getting type stability is hard here. 

function plan(w::𝕎{Tf,d,Ts}) where {d,Tf<:FFTWReal,Ts} 
	Ti = Complex{Tf}
	Si = promote_type(Tf, Ts)
	Ft = FFTW.rFFTWPlan{Tf,-1,false,d}
	It = FFTW.rFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Ts,Si,Ft,It}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 

function plan(w::𝕎{Tf,d,Ts}) where {d,Tf<:FFTWComplex,Ts} 
	Ti = Tf
	Si = promote_type(real(Tf), Ts)
	Ft = FFTW.cFFTWPlan{Tf,-1,false,d}
	It = FFTW.cFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Ts,Si,Ft,It}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 




