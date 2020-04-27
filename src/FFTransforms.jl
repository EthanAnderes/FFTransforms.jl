module FFTransforms

using Reexport
@reexport using FFTW
using AbstractFFTs
using LinearAlgebra

export plan, unitary_plan, FFT, AdjointFFT,
		𝕀, 𝕎, 𝕎32, r𝕎, r𝕎32, ⊗,
		pix, freq, rfreq 

include("xkgrids.jl")

# Aliases for FFTW type unions
# ====================================

FFTWReal    = Union{Float32,Float64}
FFTWComplex = Union{Complex{Float32},Complex{Float64}}
FFTWNumber  = Union{FFTWReal, FFTWComplex}

Plan{T,d} = Union{
	FFTW.cFFTWPlan{T,-1,false,d},
 	FFTW.rFFTWPlan{T,-1,false,d},
	FFTW.cFFTWPlan{T,1,false,d},
 	FFTW.rFFTWPlan{T,1,false,d}
}

# Holder type for forward and backward plans, region, scalars etc..
# =================================================================

struct FFT{T_forward_arg<:FFTWNumber, d, T_inverse_arg<:FFTWNumber, SF<:Number, SI<:Number, FT<:Plan, IT<:Plan}
	unscaled_forward_transform::FT
	unscaled_inverse_transform::IT
	scale_forward::SF
	scale_inverse::SI
    sz_forward_arg::NTuple{d,Int}
    sz_inverse_arg::NTuple{d,Int}
	region::NTuple{d,Bool}	

	function FFT{Tf,d}(uft::FT,uit::IT,sf::SF,si::SI,szf,szi,r) where {Tf<:FFTWNumber,d,FT,IT,SF,SI}
		Ti = Complex{real(Tf)}
		return new{Tf,d,Ti,SF,SI,FT,IT}(uft,uit,sf,si,szf,szi,r)
	end
end

struct AdjointFFT{T_forward_arg<:FFTWNumber, d, T_inverse_arg<:FFTWNumber, SF<:Number, SI<:Number, FT<:Plan, IT<:Plan}
	p::FFT{T_forward_arg, d, T_inverse_arg, SF, SI, FT, IT}
end

# Constructors
# --------------------------------
# SizeInt and RegionBool are type wrappers to allow
# @generated plan

struct SizeInt{sz} end #e.g. (512,1024,2,4)#

struct RegionBool{rg} end #e.g. (false,true,false,true)#

@generated function plan(
		::Type{T_forward_arg}, 
		::Type{SizeInt{sz_forward_arg}}, 
		::Type{RegionBool{region}},
		scale_forward::Number 
	) where {T_forward_arg<:FFTWNumber, sz_forward_arg, region}

	d          = length(sz_forward_arg)
	region_tp  = tuple(findall(region)...)
	X          = Array{T_forward_arg,d}(undef, sz_forward_arg...) 

	if T_forward_arg <: FFTWReal
		
		unscaled_forward_transform = plan_rfft(X, region_tp; flags=FFTW.ESTIMATE) 
		Y = unscaled_forward_transform * X
		unscaled_inverse_transform = plan_brfft(Y, sz_forward_arg[region_tp[1]], region_tp; flags=FFTW.ESTIMATE) 
		sz_inverse_arg = tuple(FFTW.rfft_output_size(X, region_tp)...)

	elseif T_forward_arg <: FFTWComplex

		unscaled_forward_transform = plan_fft(X, region_tp; flags=FFTW.ESTIMATE) 
		Y = unscaled_forward_transform * X
		unscaled_inverse_transform = plan_bfft(Y, region_tp; flags=FFTW.ESTIMATE) 
		sz_inverse_arg = sz_forward_arg

	end

	ifft_normalization = FFTW.normalization(real(T_forward_arg), sz_forward_arg, region_tp)

	return quote
        $(Expr(:meta, :inline))
		FFT{$T_forward_arg, $d}(
			$unscaled_forward_transform,
			$unscaled_inverse_transform,
			scale_forward,	
			$ifft_normalization / scale_forward,	
		    $sz_forward_arg,
		    $sz_inverse_arg,
			$region,	
		)
    end
end  

function Base.adjoint(p::FFT) 
	return AdjointFFT(p)
end 

function Base.adjoint(p::AdjointFFT) 
	return p.p
end 




# Define how these plan holders operate
# -------------------------------

# TODO: add mul!, lmul! and rmul!


function Base.:*(p::FFT{Tf,d}, x::Array{Tf,d}) where {d,Tf} 
	return LinearAlgebra.rmul!(p.unscaled_forward_transform * x, p.scale_forward)
end


function Base.:\(p::FFT{Tf,d,Tb}, y::Array{Tb,d}) where {d,Tf,Tb}
	return LinearAlgebra.rmul!(p.unscaled_inverse_transform * y, p.scale_inverse)
end


# the adjoint * x is the unscaled inverse transform but with forward scaling

function Base.:*(p::AdjointFFT{Tf,d,Tb}, x::Array{Tb,d}) where {d,Tf,Tb} 
	return LinearAlgebra.rmul!(p.p.unscaled_inverse_transform * x, p.p.scale_forward)
end


function Base.:\(p::AdjointFFT{Tf,d,Tb}, y::Array{Tf,d}) where {d,Tf,Tb}
	return LinearAlgebra.rmul!(p.p.unscaled_forward_transform * y, p.p.scale_inverse)
end



# Extracting/converting to a real plan and/or a complex plan
# -------------------------------

function Base.real(p::FFT{Tf,d,Ti,Sf,Si,Ft,It}) where {Tf,d,Ti,Sf,Si,Ft,It}
	Tf′ = real(Tf)
	Ft′ = FFTW.rFFTWPlan{Tf′,-1, false, d}
	It′ = FFTW.rFFTWPlan{Ti , 1, false, d}
	return plan(
		Tf′, 
		SizeInt{p.sz_forward_arg}, 
		RegionBool{p.region}, 
		p.scale_forward
	)::FFT{Tf′,d,Ti,Sf,Si,Ft′,It′}
end 

function Base.complex(p::FFT{Tf,d,Ti,Sf,Si,Ft,It}) where {Tf,d,Ti,Sf,Si,Ft,It}
	Tf′ = Complex{real(Tf)}
	Ft′ = FFTW.cFFTWPlan{Tf′,-1, false, d}
	It′ = FFTW.cFFTWPlan{Ti , 1, false, d}
	return plan(
		Tf′, 
		SizeInt{p.sz_forward_arg}, 
		RegionBool{p.region}, 
		p.scale_forward
	)::FFT{Tf′,d,Ti,Sf,Si,Ft′,It′}
end 


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



end