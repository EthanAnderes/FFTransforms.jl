using FFTransforms
using Test
FF = FFTransforms

@testset "Basic" begin

	T_forward_arg  = (Float32, Float64, Complex{Float32}, Complex{Float64})
	sz_forward_arg = ((8,7), (5,), (7,8,16), (14,4))
	region         = ((true,true), (true,), (false,true,false), (true,false))
	scale_forward =  (1, true, 1.0, 0.1f0)

	for (Tf, szf, reg, scf) in zip(T_forward_arg, sz_forward_arg, region, scale_forward)
		
		FT = plan(
			Tf, 
			FF.SizeInt{szf}, 
			FF.RegionBool{reg}, 
			scf
		)
		
		X = rand(Tf, szf)
		Y = FT.scale_forward .* (FT.unscaled_forward_transform * X)
		X′ = FT.scale_inverse .* (FT.unscaled_inverse_transform * Y)
		@test sum(abs2, X .- X′) / sum(szf) ≈ 0 atol=1e-7
	
	end

end
