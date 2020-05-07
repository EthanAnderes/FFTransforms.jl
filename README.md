# FFTransforms


Under construction ...

(Note this package defines 𝕎 which is `\BbbW<tab-complete>` in sublime but is `\bbW<tab-complete>`
in the julia REPL)








𝕎{Tf,d,Tsf,Tp} <: 𝔽{Tf,d}
ℍ{Tf,d} <: 𝔽{Tf,d}

# with these defined 
    In general with ft isa 𝔽{Tf,d}
    size_in(ft)
    size_out(ft)
    eltype_in(ft)
    eltype_out(ft)
    plan(ft) * rand(eltype_in(ft), size_in(ft))
    plan(ft) \ rand(eltype_out(ft), size_out(ft))


    Ωx(F) -> for generating white noise (grid side of the form [1,2,3,4] just contributes 1* to the Ωx )


The type must have the form Transform{Tf,To,szi,szo,...}
One can add extra parameters ... 
The transform can contain extra information like periods, transform 
scalings etc, as fields

plan(::Type{T}) where T<Transform = plan that * and div
# so the parameters of Transform{...} must know how generate it's plan










struct FFTholder{T_forward_arg<:FFTWNumber, d, T_inverse_arg<:FFTWNumber, SF<:Number, SI<:Number, FT<:Plan, IT<:Plan}
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

    