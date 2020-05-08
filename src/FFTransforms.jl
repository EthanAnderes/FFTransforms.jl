module FFTransforms

using Reexport
using LinearAlgebra
using FFTW
using AbstractFFTs
using XFields: Transform
import XFields: plan, size_in, size_out, eltype_in, eltype_out

const module_dir  = joinpath(@__DIR__, "..") |> normpath

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

# ft::𝕎{Tf,d,...} <: Transform{Tf,d}
# =========================================
# Adds a lightweight layer between a container for concrete plan 
# and the inputs to the planning methods. 
# This allows one to easily store 𝕎 as an field in an array type 
# wrapper. 

struct 𝕎{Tf<:FFTN, d, Tsf<:Number, Tp<:Real} <: Transform{Tf,d}
	sz::NTuple{d,Int} 
	region::NTuple{d,Bool}
	scale::Tsf 
	period::NTuple{d,Tp}
	function 𝕎{Tf,d}(sz::NTuple{d,Int}, rg::NTuple{d,Bool}, sc::Tsf, pd::NTuple{d,Tp}) where {Tf<:FFTN,d,Tsf,Tp}
		new{Tf,d,Tsf,Tp}(sz,rg,sc,pd)
	end
end 

@inline size_in(w::𝕎) = w.sz

size_out(w::𝕎{Tf}) where {Tf<:FFTC} = w.sz

function size_out(w::𝕎{Tf,d})::NTuple{d,Int} where {Tf<:FFTR,d}
    ir = findfirst(w.region)
    return map(w.sz, tuple(1:d...)) do nᵢ, i
        i==ir ? nᵢ÷2+1 : nᵢ
    end
end

@inline eltype_in(w::𝕎{Tf,d}) where {Tf,d}  = Tf

@inline eltype_out(w::𝕎{Tf,d}) where {Tf,d} = Complex{real(Tf)}

include("plan_fft.jl")

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTR,Tsf} 
	Ti   = Complex{Tf}
	Tsi  = promote_type(Tf, Tsf) 
	FT   = FFTW.rFFTWPlan{Tf,-1,false,d}
	IT   = FFTW.rFFTWPlan{Ti,1, false,d}
	rtn_type = FFTplan{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 

function plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTC,Tsf} 
	Ti  = Tf
	Tsi = promote_type(real(Tf), Tsf) 
	FT = FFTW.cFFTWPlan{Tf,-1,false,d}
	IT = FFTW.cFFTWPlan{Ti,1, false,d}
	rtn_type = FFTplan{Tf,d,Ti,Tsf,Tsi,FT,IT}
	return plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::rtn_type
end 

export size_in, size_out, eltype_in, eltype_out, plan, FFTplan, AdjointFFTplan


## Extra grid information available
# =====================================
# TODO: incorperate the above into a file for include so this 
# fits better in the code

include("grid.jl")

export	Δpix, Δfreq, nyq, Ωx, Ωk, 
		inv_scale, unitary_scale, ordinary_scale,
		pix, freq, fullpix, fullfreq, wavenum

#TODO: incorperate get_rFFTimpulses


## Extra convienent constructors
# =====================================
# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) untill 
# passed to the plan method

include("constructors.jl")

export	𝕀, 𝕎, 𝕎32, r𝕎, r𝕎32, ⊗, unscale𝕎, real𝕎, complex𝕎


end # Module