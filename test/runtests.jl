using FFTransforms
using Test
FF = FFTransforms



@testset "plan and adjoint (low level)" begin

	T_forward_arg  = (Float32, Float64, Complex{Float32}, Complex{Float64})
	sz_forward_arg = ((1024,1024), (5,), (7,8,16), (14,4))
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

		reg_tp = tuple(findall(reg)...) 
		if Tf <: Real
			@test sum(abs2, Y .- scf .* rfft(X,reg_tp)) / sum(szf) ≈ 0 atol=1e-7
		else
			@test sum(abs2, Y .- scf .* fft(X,reg_tp)) / sum(szf) ≈ 0 atol=1e-7
		end
	end

end




@testset "𝕀, ⊗, 𝕎, 𝕎32 and r𝕎, r𝕎32" begin

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
	@inferred plan(r𝕎32(n₁, n₂) ⊗ 𝕀(n₃, n₄))

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

	@inferred complex(𝕀(n₁) ⊗ r𝕎32(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)
	@inferred real(𝕀(n₁) ⊗ 𝕎32(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)

end



@testset "pix, fullpix, fullfreq and grid" begin

		n = 12, 10, 256, 5
		p = Float32.((1.0, 2π, 10.0, 2.0))
		r = false, true, false, true
		W1 = 𝕀(n[1]) ⊗ r𝕎(n[2]) ⊗ 𝕀(n[3]) ⊗ 𝕎(n[4])
		W2 = 𝕀(n[1]) ⊗ 𝕎(n[2]) ⊗ 𝕀(n[3]) ⊗ 𝕎(n[4])
		W3 = r𝕎32(n[1]) ⊗ 𝕎32(n[4])
		W4 = 𝕎32(n[2], n[4])

		@inferred pix(n[1], p[1])
		@inferred pix(n, p)
		@inferred freq(n[1], p[1])
		@inferred freq(n, p)
		@inferred freq(n, p, r)
		@inferred FFTransforms.rfreq(n, p, r)

		# TODO: make these type stable
		grid(W1, p)
		grid(W2, p)
		grid(W3, (p[1], p[4]))
		grid(W4, (p[1], p[4]))

		@inferred pix(W1, p)
		@inferred pix(W2, p)
		@inferred pix(W3, (p[1], p[4]))
		@inferred pix(W4, (p[2], p[4]))

		@inferred freq(W1, p)
		@inferred freq(W2, p)
		@inferred freq(W3, (p[1], p[4]))
		@inferred freq(W4, (p[2], p[4]))

		@inferred fullpix(1, W1, p)
		@inferred fullpix(2, W2, p)
		@inferred fullpix(W1, p)
		@inferred fullpix(W2, p)

		@inferred fullfreq(1, W1, p)
		@inferred fullfreq(2, W2, p)
		@inferred fullfreq(W1, p)
		@inferred fullfreq(W2, p)

		@inferred wavenum(W1, p)
		@inferred wavenum(W2, p)

end