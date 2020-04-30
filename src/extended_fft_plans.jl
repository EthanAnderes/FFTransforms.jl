
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
	flags      = FFTW.ESTIMATE #fixme: the other ones don't seem to work here 
	timelim    = 10.0

	if T_forward_arg <: FFTWReal
		
		T_backward_arg = Complex{T_forward_arg}
		sz_inverse_arg = tuple(FFTW.rfft_output_size(X, region_tp)...)
		Y = Array{T_backward_arg,d}(undef, sz_inverse_arg...)
		unscaled_forward_transform = FFTW.rFFTWPlan{T_forward_arg, -1,false,d}(X, Y, region_tp, flags, timelim)
		unscaled_inverse_transform = FFTW.rFFTWPlan{T_backward_arg, 1,false,d}(Y, X, region_tp, flags, timelim)

	elseif T_forward_arg <: FFTWComplex

		sz_inverse_arg = sz_forward_arg
		Y = Array{T_forward_arg,d}(undef, sz_inverse_arg...)
		unscaled_forward_transform = FFTW.cFFTWPlan{T_forward_arg,-1,false,d}(X, Y, region_tp, flags, timelim)
		unscaled_inverse_transform = FFTW.cFFTWPlan{T_forward_arg, 1,false,d}(Y, X, region_tp, flags, timelim)

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


