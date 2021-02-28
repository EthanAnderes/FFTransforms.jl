module FFTransforms

using LinearAlgebra
using FFTW
using XFields

const module_dir  = joinpath(@__DIR__, "..") |> normpath

C64 = Complex{Float64}
C32 = Complex{Float32}
F64 = Float64
F32 = Float32

FFTR = Union{F32,F64}
FFTC = Union{C32,C64}
FFTN = Union{FFTR, FFTC}

Plan{T,d,G} = Union{
	FFTW.cFFTWPlan{T,-1,false,d,G},
 	FFTW.rFFTWPlan{T,-1,false,d,G},
	FFTW.cFFTWPlan{T,1,false,d,G},
 	FFTW.rFFTWPlan{T,1,false,d,G}
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

@inline XFields.size_in(w::𝕎) = w.sz

XFields.size_out(w::𝕎{Tf}) where {Tf<:FFTC} = w.sz

function XFields.size_out(w::𝕎{Tf,d})::NTuple{d,Int} where {Tf<:FFTR,d}
    ir = findfirst(w.region)
    return map(w.sz, tuple(1:d...)) do nᵢ, i
        i==ir ? nᵢ÷2+1 : nᵢ
    end
end

@inline XFields.eltype_in(w::𝕎{Tf,d}) where {Tf,d}  = Tf

@inline XFields.eltype_out(w::𝕎{Tf,d}) where {Tf,d} = Complex{real(Tf)}

include("plan_fft.jl")

function XFields.plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTR,Tsf} 
	Ti   = Complex{Tf}
	Tsi  = promote_type(Tf, Tsf)
	plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::FFTplan{Tf,d,Ti,Tsf,Tsi}
end 

function XFields.plan(w::𝕎{Tf,d,Tsf}) where {d,Tf<:FFTC,Tsf} 
	Ti  = Tf
	Tsi = promote_type(real(Tf), Tsf)
	plan(Tf,SizeInt{w.sz},RegionBool{w.region},w.scale)::FFTplan{Tf,d,Ti,Tsf,Tsi}
end 

export size_in, size_out, eltype_in, eltype_out, plan, FFTplan, AdjointFFTplan


## Extra grid information available
# =====================================
# TODO: incorperate the above into a file for include so this 
# fits better in the code

include("grid.jl")

# export	Δpix, Δfreq, nyq, Ωpix, Ωfreq, 
# 		inv_scale, unitary_scale, ordinary_scale,
# 		pix, freq, fullpix, fullfreq, wavenum

#TODO: incorperate get_rFFTimpulses


## rand_in, rand_out (incomplete), dot_in, dot_out (incomplete)
# =====================================

include("methods.jl")


## Extra convienent constructors
# =====================================
# The following structs allow lazy construction of an fft plan.
# Mixing 𝕀 and 𝕎 with ⊗ creates another 𝕎 (or 𝕀) untill 
# passed to the plan method

include("constructors.jl")

export 𝕀, 𝕎, ⊗ #, unscale, real, complex


end # Module