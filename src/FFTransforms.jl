module FFTransforms

using Reexport
@reexport using FFTW
using AbstractFFTs
using LinearAlgebra

export plan

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
# if FS == Bool(true) then no scale is preformed

struct FFT{d, T_forward_arg<:FFTWNumber, T_inverse_arg<:FFTWNumber, SF<:Number, SI<:Number, FT<:Plan, IT<:Plan}
	unscaled_forward_transform::FT
	unscaled_inverse_transform::IT
	scale_forward::SF	
	scale_inverse::SI	
    sz_forward_arg::NTuple{d,Int}
    sz_inverse_arg::NTuple{d,Int}
	region::NTuple{d,Bool}	
end

struct SizeInt{sz} end #(512,1024,2,4)#

struct RegionBool{rg} end #(false,true,false,1)#


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
		unscaled_inverse_transform = plan_brfft(Y, sz_forward_arg[1], region_tp; flags=FFTW.ESTIMATE) 
		sz_inverse_arg = tuple(FFTW.rfft_output_size(X, region_tp)...)

	elseif T_forward_arg <: FFTWComplex

		unscaled_forward_transform = plan_fft(X, region_tp; flags=FFTW.ESTIMATE) 
		Y = unscaled_forward_transform * X
		unscaled_inverse_transform = plan_bfft(Y, region_tp; flags=FFTW.ESTIMATE) 
		sz_inverse_arg = sz_forward_arg

	end

	FT = typeof(unscaled_forward_transform)
	IT = typeof(unscaled_inverse_transform)
	T_inverse_arg = eltype(Y)

	ifft_normalization = FFTW.normalization(X, region_tp)

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




# 𝕀{sz} ⊗ 𝕎{sz,T}

# 𝕎{1}()
# FT{T,n}

# Id{n}



# BT

# 2*FTf(n2)
# 2*FTf(n2)  

# I(200) ⊗ FTf(1024) ⊗ I(2)
# FTf(1024) ⊗ I(200) ⊗ I(2)

# I(n1) ⊗ FTf(n2) ⊗ I(n3)
# I(n1) ⊗ FTf(T,n2,scale) ⊗ I(n3)


# Fft{T}(isz::NTuple{Int})        -> Fft{T,length(isz), Bool, T == Real ? rFFTW : cFFTW }
# Fft{T}(isz::NTuple{Int}, scale) -> Fft{T,length(isz), T, T == Real ? rFFTW : cFFTW }

# IFT::FFTW.rFFTWPlan{Complex{T},1,false,d}


end