

# plans via kron's of FFTs and Identity operators
# ================================================

# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) until a final
# scalar multplier on the right triggers plan creation

# 𝕎 holds sz (size of the input array) 
# and region (which determines which axes get FFT'd)
# --------------------------
struct 𝕎{T_forward_arg<:FFTWNumber,d}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool} 
end 

𝕎(::Type{T}, n::Vararg{Int,d}) where {T,d} = 𝕎{T,d}(n, tuple(trues(d)...))

# some shorthand alternatives to 𝕎(n) which defaults to Complex{Float64}

𝕎(n::Vararg{Int,d}) where {d} = 𝕎{Complex{Float64},d}(n, tuple(trues(d)...))

𝕎32(n::Vararg{Int,d}) where {d} = 𝕎{Complex{Float32},d}(n, tuple(trues(d)...))

r𝕎(n::Vararg{Int,d}) where {d} = 𝕎{Float64,d}(n, tuple(trues(d)...))

r𝕎32(n::Vararg{Int,d}) where {d} = 𝕎{Float32,d}(n, tuple(trues(d)...))


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

function ⊗(i::𝕀{n}, w::𝕎{T,d}) where {n,T,d} 
	sz     = tuple(i.sz..., w.sz...)
	region = tuple(falses(n)..., w.region...)
	return 𝕎{T,d+n}(sz,region)
end

function ⊗(w::𝕎{T,d}, i::𝕀{n}) where {n,T,d} 
	sz     = tuple(w.sz..., i.sz...)
	region = tuple(w.region..., falses(n)...)
	return 𝕎{T,d+n}(sz,region)
end

# The element type of the first 𝕎 (reading left to right) determines 
# the overall type of the transform 
# Do we want this to promote on the real type? 
# .... so 𝕎{R<:FFTWReal,d}    ⊗ 𝕎{T,d} -> promote_type(R,real(T))
# ....and 𝕎{R<:FFTWComplex,d} ⊗ 𝕎{T,d} -> Complex{promote_type(real(R),real(T))}
function ⊗(w::𝕎{R,d}, v::𝕎{T,n}) where {d,n,R<:FFTWNumber,T<:FFTWNumber} 
	sz     = tuple(w.sz..., v.sz...)
	region = tuple(w.region..., v.region...)
	return 𝕎{R,d+n}(sz,region)
end


# Triggering a planned FFT from 𝕎 
# ===========================================

# Scalar multiply on the right is the trigger for generating a concrete plan
# scale == true is the scentanal for an unscaled plan
# 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₁) ⊗ 𝕎(n₂) -> 𝕎
# 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₁) ⊗ 𝕎(n₂) * scale -> plan

Base.:*(w::𝕎{T,d}, s::S) where {d,T,S} = plan(w, s)

plan(w::𝕎{T,d}) where {T,d} = plan(w::𝕎{T,d}, true) 

function unitary_plan(w::𝕎{T,d}) where {T,d}
	s = prod(1/√i[1] for i in zip(w.sz,w.region) if i[2])
	return plan(w, s)
end


function plan(w::𝕎{Tf,d}, s::S) where {d,Tf,S} 
	real(T_forward_arg)
	return plan(T,SizeInt{w.sz},RegionBool{w.region},s)
end 


# fixme: Getting type stability is hard here. 

function plan(w::𝕎{Tf,d}, s::Sf) where {d,Tf<:FFTWReal,Sf} 
	Ti = Complex{Tf}
	Si = promote_type(Tf, Sf)
	Ft = FFTW.rFFTWPlan{Tf,-1,false,d}
	It = FFTW.rFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Sf,Si,Ft,It}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},s)::rtn_type
end 

function plan(w::𝕎{Tf,d}, s::Sf) where {d,Tf<:FFTWComplex,Sf} 
	Ti = Tf
	Si = promote_type(real(Tf), Sf)
	Ft = FFTW.cFFTWPlan{Tf,-1,false,d}
	It = FFTW.cFFTWPlan{Ti,1, false,d}
	rtn_type = FFT{Tf,d,Ti,Sf,Si,Ft,It}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},s)::rtn_type
end 




