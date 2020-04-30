module FFTransforms

using Reexport
@reexport using FFTW
using AbstractFFTs
using LinearAlgebra


# Aliases for FFTW eltypes
# ---------------------------------
C64 = Complex{Float64}
C32 = Complex{Float32}
F64 = Float64
F32 = Float32

FFTR = Union{F32,F64}
FFTC = Union{C32,C64}
FFTN = Union{FFTR, FFTC}

Plan{T,d} = Union{
	FFTW.cFFTWPlan{T,-1,false,d},
 	FFTW.rFFTWPlan{T,-1,false,d},
	FFTW.cFFTWPlan{T,1,false,d},
 	FFTW.rFFTWPlan{T,1,false,d}
}

# 
# ---------------------------------
	
include("extended_fft_plans.jl")
export plan, FFTplan, AdjointFFTplan

include("w_kron_id.jl")
export	𝕀, 𝕎, 𝕎32, r𝕎, r𝕎32, ⊗,
		unscale𝕎, real𝕎, complex𝕎
		
#TODO: incorperate the above into a file for include so this 
# fits better in the code
include("xkgrids.jl")
export	size_in, size_out, eltype_in, eltype_out,
		Δpix, Δfreq, nyq, Ωx, Ωk, 
		inv_scale, unitary_scale, ordinary_scale,
		pix, freq, fullpix, fullfreq, wavenum



end # Module