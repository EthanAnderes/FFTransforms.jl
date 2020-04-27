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
			# FT = plan(
			# 	Tf, 
			# 	szf, 
			# 	reg, 
			# 	scf
			# )



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

		FT1 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true
		FT2 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 10.0
		FT3 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 2
		FT4 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> unitary_plan
		FT5 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> plan
		FT6 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> plan |> adjoint
		FT7 = FT5'
		FT8 = plan(r𝕎(n₁, n₂) ⊗ 𝕀(n₃, n₄))'
		@inferred plan(r𝕎(n₁, n₂) ⊗ 𝕀(n₃, n₄))

		X = rand(Float64, n₁, n₂, n₃, n₄)


		for FT ∈ (FT1, FT2, FT3, FT4, FT5)
			@inferred FT \ (FT * X)
		end

		for FT ∈ (FT6, FT7, FT8)
			@test FT' isa FFT
			Y = FT' * X
			@inferred FT \ (FT * Y)
		end

		@inferred complex(𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)
		@inferred real(𝕀(n₁) ⊗ 𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)

	end


	let 
		n₁, n₂, n₃, n₄ = 12, 10, 256, 5
		p₁, p₂, p₃, p₄ = 1.0, 2π, 10.0, 2.0
		r₁, r₂, r₃, r₄ = false, true, false, true
		@inferred pix(n₁, p₁)
		@inferred pix((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄))
		@inferred freq(n₁, p₁)
		@inferred freq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄))
		@inferred freq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄), (r₁, r₂, r₃, r₄))
		@inferred rfreq((n₁, n₂, n₃, n₄), (p₁, p₂, p₃, p₄), (r₁, r₂, r₃, r₄))
	end


end
