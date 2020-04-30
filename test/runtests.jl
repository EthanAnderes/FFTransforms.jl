using FFTransforms
using Test
FF = FFTransforms




@testset "𝕎  constructors" begin 


	# 1-d constructors
	# ------------------
	ns  = [7, 12] 

	Ia  = 𝕀(ns[1])  
	Ib  = 𝕀(ns[2])
	Ic  = 𝕀(ns[1], 2π)
	Id  = 𝕀(ns[2], 2π)  

	Wa  = 𝕎(ns[1])  
	Wb  = 𝕎(ns[2])     * 2.0  
	Wc  = 𝕎(ns[1], 2π) * 2.0f0 
	Wd  = 𝕎(ns[2], 2π)  

	W32a  = 𝕎32(ns[1])  
	W32b  = 𝕎32(ns[2]) * 2.0  
	W32c  = 𝕎32(ns[1], 2π) * 2.0f0 
	W32d  = 𝕎32(ns[2], 2π)  


	rWa  = r𝕎(ns[1])  
	rWb  = r𝕎(ns[2]) * 2.0  
	rWc  = r𝕎(ns[1], 2π) * 2.0f0 
	rWd  = r𝕎(ns[2], 2π)  

	rW32a  = 𝕎32(ns[1])  
	rW32b  = 𝕎32(ns[2]) * 2.0  
	rW32c  = 𝕎32(ns[1], 2π) * 2.0f0 
	rW32d  = 𝕎32(ns[2], 2π)  


	# kron of 1-d 
	# ------------------

	W = Ia ⊗ Wa ⊗ rW32d
	W = Ib ⊗ rW32a ⊗ Wb
	W = Ic ⊗ rW32a ⊗ Wb
	W = rW32a ⊗ Wb ⊗ Ic


	szi  = @inferred size_in(W)
	szo  = @inferred size_out(W)
	WTf  = @inferred eltype_in(W)
	WTi  = @inferred eltype_out(W)
	Δx   = @inferred Δpix(W)
	Δk   = @inferred Δfreq(W)
	nq   = @inferred nyq(W)
	ωx   = @inferred Ωx(W)
	ωk   = @inferred Ωk(W) 
	invs = @inferred inv_scale(W)
	us   = @inferred unitary_scale(W)
	os   = @inferred ordinary_scale(W)
	xvecs   = @inferred pix(W)
	kvecs   = @inferred freq(W)
	xarrays = @inferred fullpix(W)
	karrays = @inferred fullfreq(W)
	λmat    = @inferred wavenum(W)

	sW = @inferred W * (1/√(2π))
	uW = @inferred unitary_scale(W) * W
	oW = @inferred ordinary_scale(W) * W
	W′ = @inferred unscale𝕎(sW)
	rW = @inferred real𝕎(W)
	cW = @inferred complex𝕎(W)


	P  = @inferred plan(W)
	sP = @inferred plan(sW)
	uP = @inferred plan(uW)
	oP = @inferred plan(oW)
	P′ = @inferred plan(W′)
	rP = @inferred plan(rW)
	cP = @inferred plan(cW)


	Pout  = P  * rand(eltype_in(W), size_in(W))
	sPout = sP * rand(eltype_in(sW), size_in(sW))
	uPout = uP * rand(eltype_in(uW), size_in(uW))
	oPout = oP * rand(eltype_in(oW), size_in(oW))
	P′out = P′ * rand(eltype_in(W′), size_in(W′))
	rPout = rP * rand(eltype_in(rW), size_in(rW))
	cPout = cP * rand(eltype_in(cW), size_in(cW))

	P  \ Pout 
	sP \ sPout
	uP \ uPout
	oP \ oPout
	P′ \ P′out
	rP \ rPout
	cP \ cPout


end






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






# @testset "𝕀, ⊗, 𝕎, 𝕎32 and r𝕎, r𝕎32" begin

# 	n₁, n₂, n₃, n₄ = 12, 10, 256, 2
	
# 	W1 = 𝕀(n₁)
# 	W2 = 𝕀(n₁) ⊗ r𝕎(n₂)
# 	W3 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃)
# 	W4 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄)
# 	W5 = 𝕀(n₁, n₂)   ⊗ 𝕎(n₃)
# 	W6 = r𝕎(n₁,n₂)  ⊗ 𝕀(n₃,n₄)

# 	FT1 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true
# 	FT2 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 10.0
# 	FT3 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * 2
# 	FT4 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> unitary_plan
# 	FT5 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> plan
# 	FT6 = 𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) |> plan |> adjoint
# 	FT7 = FT5'
# 	FT8 = plan(r𝕎(n₁, n₂) ⊗ 𝕀(n₃, n₄))'
# 	@inferred plan(r𝕎32(n₁, n₂) ⊗ 𝕀(n₃, n₄))

# 	X = rand(Float64, n₁, n₂, n₃, n₄)


# 	for FT ∈ (FT1, FT2, FT3, FT4, FT5)
# 		@inferred FT \ (FT * X)
# 	end

# 	for FT ∈ (FT6, FT7, FT8)
# 		@test FT' isa FFT
# 		Y = FT' * X
# 		@inferred FT \ (FT * Y)
# 	end

# 	@inferred complex(𝕀(n₁) ⊗ r𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)
# 	@inferred real(𝕀(n₁) ⊗ 𝕎(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)

# 	@inferred complex(𝕀(n₁) ⊗ r𝕎32(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)
# 	@inferred real(𝕀(n₁) ⊗ 𝕎32(n₂) ⊗ 𝕀(n₃) ⊗ 𝕎(n₄) * true)

# end

