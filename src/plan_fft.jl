
# TODO: writeup documentation

# Holder type for forward and backward plans, region, scalars etc..
# =================================================================
# Tf ≡ T_forward_arg
# Ti ≡ T_inverse_arg


struct FFTplan{Tf<:FFTN, d, Ti<:FFTN, Tsf<:Number, Tsi<:Number, FT<:Plan, IT<:Plan}
	unscaled_forward_transform::FT
	unscaled_inverse_transform::IT
	scale_forward::Tsf
	scale_inverse::Tsi
    sz_forward_arg::NTuple{d,Int}
    sz_inverse_arg::NTuple{d,Int}
	region::NTuple{d,Bool}	

	function FFTplan{Tf,d}(uft::FT,uit::IT,sf::Tsf,si::Tsi,szf,szi,r) where {Tf<:FFTN,d,FT,IT,Tsf,Tsi}
		Ti = Complex{real(Tf)}
		return new{Tf,d,Ti,Tsf,Tsi,FT,IT}(uft,uit,sf,si,szf,szi,r)
	end
end

struct AdjointFFTplan{Tf<:FFTN, d, Ti<:FFTN, Tsf<:Number, Tsi<:Number, FT<:Plan, IT<:Plan}
	p::FFTplan{Tf, d, Ti, Tsf, Tsi, FT, IT}
end

# Constructors
# --------------------------------
# SizeInt and RegionBool are type wrappers to allow
# @generated plan

struct SizeInt{sz} end #e.g. (512,1024,2,4)#

struct RegionBool{rg} end #e.g. (false,true,false,true)#

@generated function plan(
		::Type{Tf}, 
		::Type{SizeInt{sz_forward_arg}}, 
		::Type{RegionBool{region}},
		scale_forward::Number 
	) where {Tf<:FFTN, sz_forward_arg, region}

	d          = length(sz_forward_arg)
	region_tp  = tuple(findall(region)...)
	X          = Array{Tf,d}(undef, sz_forward_arg...) 
	flags      = FFTW.ESTIMATE #fixme: the other ones don't seem to work here 
	timelim    = 20.0

	if Tf <: FFTR
		
		T_backward_arg = Complex{Tf}
		sz_inverse_arg = tuple(FFTW.rfft_output_size(X, region_tp)...)
		Y = Array{T_backward_arg,d}(undef, sz_inverse_arg...)
		unscaled_forward_transform = FFTW.rFFTWPlan{Tf, -1,false,d}(X, Y, region_tp, flags, timelim)
		unscaled_inverse_transform = FFTW.rFFTWPlan{T_backward_arg, 1,false,d}(Y, X, region_tp, flags, timelim)

	elseif Tf <: FFTC

		sz_inverse_arg = sz_forward_arg
		Y = Array{Tf,d}(undef, sz_inverse_arg...)
		unscaled_forward_transform = FFTW.cFFTWPlan{Tf,-1,false,d}(X, Y, region_tp, flags, timelim)
		unscaled_inverse_transform = FFTW.cFFTWPlan{Tf, 1,false,d}(Y, X, region_tp, flags, timelim)

	end

	ifft_normalization = FFTW.normalization(real(Tf), sz_forward_arg, region_tp)

	return quote
 	       $(Expr(:meta, :inline))
 	       FFTplan{$Tf, $d}(
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

function Base.adjoint(p::FFTplan) 
	return AdjointFFT(p)
end 

function Base.adjoint(p::AdjointFFTplan) 
	return p.p
end 




# Define how these plan holders operate
# -------------------------------

function LinearAlgebra.mul!(y::Array{Ti,d}, p::FFTplan{Tf,d,Ti}, x::Array{Tf,d}) where {d,Tf,Ti}
	mul!(y, p.unscaled_forward_transform, x)
	## rmul!(y, p.scale_forward)
	@inbounds y .*= p.scale_forward
	return y
end


function LinearAlgebra.ldiv!(x::Array{Tf,d}, p::FFTplan{Tf,d,Ti}, y::Array{Ti,d}) where {d,Tf,Ti}
	mul!(x, p.unscaled_inverse_transform, y)
	## rmul!(x, p.scale_inverse)
	@inbounds x .*= p.scale_inverse
	return x
end


function Base.:*(p::FFTplan{Tf,d}, x::Array{Tf,d}) where {d,Tf} 
	## return LinearAlgebra.rmul!(p.unscaled_forward_transform * x, p.scale_forward)
	rtn = p.unscaled_forward_transform * x
	@inbounds rtn .*= p.scale_forward
	return rtn
end


function Base.:\(p::FFTplan{Tf,d,Ti}, y::Array{Ti,d}) where {d,Tf,Ti}
	## return LinearAlgebra.rmul!(p.unscaled_inverse_transform * y, p.scale_inverse)
	rtn   = p.unscaled_inverse_transform * y
	@inbounds rtn .*= p.scale_inverse
	return rtn 
end


# the adjoint * x is the unscaled inverse transform but with forward scaling

function Base.:*(p::AdjointFFTplan{Tf,d,Ti}, x::Array{Ti,d}) where {d,Tf,Ti} 
	## return LinearAlgebra.rmul!(p.p.unscaled_inverse_transform * x, p.p.scale_forward)
	rtn = p.p.unscaled_inverse_transform * x
	@inbounds rtn .*= p.p.scale_forward
end


function Base.:\(p::AdjointFFTplan{Tf,d,Ti}, y::Array{Tf,d}) where {d,Tf,Ti}
	## return LinearAlgebra.rmul!(p.p.unscaled_forward_transform * y, p.p.scale_inverse)
	rtn   = p.p.unscaled_forward_transform * y
	@inbounds rtn .*= p.p.scale_inverse
	return rtn
end



# Extracting/converting to a real plan and/or a complex plan
# -------------------------------

function Base.real(p::FFTplan{Tf,d,Ti,Sf,Si,FT,IT}) where {Tf,d,Ti,Sf,Si,FT,IT}
	Tf′ = real(Tf)
	G    = NTuple{sum(p.region),Int} 
	FT′ = FFTW.rFFTWPlan{Tf′,-1, false, d, G}
	IT′ = FFTW.rFFTWPlan{Ti , 1, false, d, G}
	return plan(
		Tf′, 
		SizeInt{p.sz_forward_arg}, 
		RegionBool{p.region}, 
		p.scale_forward
	)::FFTplan{Tf′,d,Ti,Sf,Si,FT′,IT′}
end 

function Base.complex(p::FFTplan{Tf,d,Ti,Sf,Si,FT,IT}) where {Tf,d,Ti,Sf,Si,FT,IT}
	Tf′ = Complex{real(Tf)}
	G   = NTuple{sum(p.region),Int} 
	FT′ = FFTW.cFFTWPlan{Tf′,-1, false, d, G}
	IT′ = FFTW.cFFTWPlan{Ti , 1, false, d, G}
	return plan(
		Tf′, 
		SizeInt{p.sz_forward_arg}, 
		RegionBool{p.region}, 
		p.scale_forward
	)::FFTplan{Tf′,d,Ti,Sf,Si,FT′,IT′}
end 


