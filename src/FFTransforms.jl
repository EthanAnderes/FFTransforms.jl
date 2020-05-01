module FFTransforms

using Reexport
@reexport using FFTW
using AbstractFFTs
using LinearAlgebra

#TODO change to XFields once ready
using CMBrings 
import CMBrings: plan, size_in, size_out, eltype_in, eltype_out, Ωx




# import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
# import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# Aliases for FFTW eltypes
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

# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) untill 
# passed to the plan method

# 𝕎 holds sz (size of the input array) 
# and region (which determines which axes get FFT'd)
# =====================================
struct 𝕎{Tf<:FFTN, d, Tsf<:Number, Tp<:Real} <: Transform{Tf,d}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool}
	scale::Tsf 
	period::NTuple{d,Tp}

	function 𝕎{Tf,d}(sz::NTuple{d,Int}, rg::NTuple{d,Bool}, sc::Tsf, pd::NTuple{d,Tp}) where {Tf<:FFTN,d,Tsf,Tp}
		new{Tf,d,Tsf,Tp}(sz,rg,sc,pd)
	end
end 


include("extended_fft_plans.jl")
export plan, FFTplan, AdjointFFTplan, 
		eltype_in, eltype_out


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