using FFTransforms
using Test
FF = FFTransforms

@testset "Basic" begin

	let 
		T_forward_arg  = (Float32, Float64, Complex{Float32}, Complex{Float64})
		sz_forward_arg = ((8,7), (5,), (7,8,16), (14,4))
		region         = ((false,true), (true,), (false,true,false), (true,false))
		scale_forward =  (1, true, 1.0, 0.1f0)

		for indx = 1:length(T_forward_arg)
			Tf  = T_forward_arg[indx]
			szf = sz_forward_arg[indx]
			reg = region[indx]
			scf = scale_forward[indx]
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



	let 
		n₁, n₂, n₃, n₄ = 12, 10, 256, 2
		
		W1 = 𝕀(n₁)
		W2 = 𝕀(n₁) ⊗ r𝕎(n₂)
		W3 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃)
		W4 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄)
		W5 = 𝕀(n₁, n₂)   ⊗ 𝕎(n₃)
		W6 = r𝕎(n₁,n₂)  ⊗ 𝕀(n₃,n₄)
		# W7 = r𝕎(n₁,n₂)' ⊗ 𝕀(n₃,n₄)

		FT1 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true
		FT2 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 10.0
		FT3 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 2
		#FT4 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> unitary_plan
		#FT5 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> plan

		X = rand(Float64, n₁, n₂, n₃, n₄)
		FT1 \ (FT1 * X)
		FT2 \ (FT2 * X)
		FT3 \ (FT3 * X)
	end


	let 
		n₁, n₂, n₃, n₄ = 12, 10, 256, 5
		p₁, p₂, p₃, p₄ = 1.0, 2π, 10.0, 2.0
		r₁, r₂, r₃, r₄ = false, true, false, true
		pix(n₁, p₁)
		pix((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄))
		freq(n₁, p₁)
		freq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄))
		freq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄), (r₁, r₂, r₃, r₄))
		rfreq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄), (r₁, r₂, r₃, r₄))
	end


	# perhaps fields are xmap{W,scale}

end
