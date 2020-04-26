module FFTransforms

using Reexport
@reexport using FFTW
using AbstractFFTs
using LinearAlgebra

export plan, 𝕎, r𝕎, 𝕀, ⊗
export pix, freq, rfreq 

# this needs updating for these split plans
include("xkgrids.jl")

FFTWReal    = Union{Float32,Float64}
FFTWComplex = Union{Complex{Float32},Complex{Float64}}
FFTWNumber  = Union{FFTWReal, FFTWComplex}

Plan{T,d} = Union{
	FFTW.cFFTWPlan{T,-1,false,d},
 	FFTW.rFFTWPlan{T,-1,false,d},
	FFTW.cFFTWPlan{T,1,false,d},
 	FFTW.rFFTWPlan{T,1,false,d}
}

# completely describes a plan 
# if scale_forward[i] == true then scale is one and no mult is preformed

# TODO: make another struct which holds it's own storage

struct FFT{d, T_forward_arg<:FFTWNumber, T_inverse_arg<:FFTWNumber, SF<:Number, SI<:Number, FT<:Plan, IT<:Plan}
	unscaled_forward_transform::FT
	unscaled_inverse_transform::IT
	scale_forward::SF	
	scale_inverse::SI	
    sz_forward_arg::NTuple{d,Int}
    sz_inverse_arg::NTuple{d,Int}
	region::NTuple{d,Bool}	
end

struct SizeInt{sz} end #e.g. (512,1024,2,4)#

struct RegionBool{rg} end #e.g. (false,true,false,true)#

@generated function plan(
		::Type{T_forward_arg}, 
		::Type{SizeInt{sz_forward_arg}}, 
		::Type{RegionBool{region}}, 
		scale_forward::SF
	) where {T_forward_arg<:FFTWNumber, sz_forward_arg, region, SF<:Number}

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

	FT = typeof(unscaled_forward_transform)
	IT = typeof(unscaled_inverse_transform)

	real_T_inverse_arg = real(T_forward_arg)
	T_inverse_arg = Complex{real_T_inverse_arg} 

	ifft_normalization = FFTW.normalization(real_T_inverse_arg, sz_forward_arg, region_tp)

	return quote
        $(Expr(:meta, :inline))
        scale_inverse  = $ifft_normalization / scale_forward
		SI = typeof(scale_inverse)
		FFT{$d, T_forward_arg, $T_inverse_arg, SF, SI, $FT, $IT}(
			$unscaled_forward_transform,
			$unscaled_inverse_transform,
			scale_forward,	
			scale_inverse,	
		    sz_forward_arg,
		    $sz_inverse_arg,
			region,	
		)
    end
end  


# TODO: plan(𝕎{d,T}) ...
# TODO: unitary_plan(𝕎{d,T}) ... 
# TODO: adjoint plan 

# Define how these plan holders operate
# todo mul! and adjoint
# -------------------------------

function Base.:*(p::FFT{d,Tf}, x::Array{Tf,d}) where {d,Tf} 
	#return p.scale_forward .* (p.unscaled_forward_transform * x)
	return LinearAlgebra.rmul!(p.unscaled_forward_transform * x, p.scale_forward)
end


function Base.:\(p::FFT{d,Tf,Tb}, y::Array{Tb,d}) where {d,Tf,Tb}
	#return p.scale_inverse .* (p.unscaled_inverse_transform * y)
	return LinearAlgebra.rmul!(p.unscaled_inverse_transform * y, p.scale_inverse)
end


# plans via kron's of FFTs and Identity operators
# ---------------------

# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) until a final
# scalar multplier on the right triggers plan creation

struct 𝕎{d, T_forward_arg<:FFTWNumber}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool} 
end 

𝕎(n::Vararg{Int,d})            where d     = 𝕎{d,Complex{Float64}}(n, tuple(trues(d)...))
𝕎(::Type{T}, n::Vararg{Int,d}) where {d,T} = 𝕎{d,T}(n, tuple(trues(d)...))

r𝕎(n::Vararg{Int,d})            where d     = 𝕎{d,Float64}(n, tuple(trues(d)...))
r𝕎(::Type{T}, n::Vararg{Int,d}) where {d,T} = 𝕎{d,T}(n, tuple(trues(d)...))

struct 𝕀{d}
	sz::NTuple{d,Int} 
end 

𝕀(n::Vararg{Int,d}) where d = 𝕀{d}(n)


# Define the lazy kron operators. The last mult on the right by a scalar 
# is the trigger for generating a concrete plan
function ⊗(i::𝕀{n}, j::𝕀{d}) where {n,d} 
	sz  = tuple(i.sz..., j.sz...)
	return 𝕀{d+n}(sz)
end

function ⊗(i::𝕀{n}, w::𝕎{d,T}) where {n,d,T} 
	sz     = tuple(i.sz..., w.sz...)
	region = tuple(falses(n)..., w.region...)
	return 𝕎{d+n,T}(sz,region)
end

function ⊗(w::𝕎{d,T}, i::𝕀{n}) where {n,d,T} 
	sz     = tuple(w.sz..., i.sz...)
	region = tuple(w.region..., falses(n)...)
	return 𝕎{d+n,T}(sz,region)
end

# The element type of the first 𝕎 (reading left to right) determines 
# the overall type of the transform 
function ⊗(w::𝕎{d,R}, v::𝕎{n,T}) where {d,n,R<:FFTWReal,T<:FFTWNumber} 
	sz     = tuple(w.sz..., v.sz...)
	region = tuple(w.region..., v.region...)
	return 𝕎{d+n,R}(sz,region)
end

# Scalar multiply on the right is the trigger for generating a concrete plan
# scale == true is the scentanal for an unscaled plan
# 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₁) ⊗ 𝕎(n₂) -> 𝕎
# 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₁) ⊗ 𝕎(n₂) * scale -> plan
function Base.:*(w::𝕎{d,T}, s::S) where {d,T,S} 
	plan(T,SizeInt{w.sz},RegionBool{w.region},s)
end 



end