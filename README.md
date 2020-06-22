# FFTransforms

Under construction ...

### FFT Transform type for XFields


(Note this package defines 𝕎 which is `\BbbW<tab-complete>` in sublime but is `\bbW<tab-complete>`in the julia REPL)


# Quick start




# Required methods to hook into XFields ...


```
struct NewTransform{Tf,d,...} <: Transform{Tf,d}
    <any fields here necessary for determining the transform>
end
```

* `size_in(nT::NewTransform) -> <size of the storage for the corresponding MapField>`
* `size_out(nT::NewTransform) -> <size of the storage for the corresponding FourierField>`  
* `eltype_in(nT::NewTransform) -> <eltype of the storage field of the corresponding Field>`
* `eltype_out(nT::NewTransform) -> <eltype of the storage for the corresponding FourierField>`
* `plan(nT::NewTransform) * <storage for the corresponding MapField>`
* `plan(nT::NewTransform) \ <storage for the corresponding FourierField>`


Note: if the transform requires custom methods to convert Map <-> Fourier then one can simply define `plan(nT::NewTransform) = nT` and follow up with overloading `*(nT,<storage>)` and `*(nT,<storage>)` for  `nT::NewTransform`.

