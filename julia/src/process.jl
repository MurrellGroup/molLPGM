Base.@kwdef struct MolecularPrepConfig
    coordinate_process::String = "fragment-se3-plus-internal"
    atom_bm_variance::Float64 = 14.8
    atom_bm_clock_a::Float64 = 2.3
    atom_bm_clock_b::Float64 = 3.1
    token_capacity::Int = 4096
    fragment_capacity::Int = 1365
    max_documents::Int = 512
    source_sigma::Float32 = 3f0
    center_variance::Float32 = 100f0 / 9f0
    center_spread_sigma::Float32 = 3f0
    center_spread_timescale::Float32 = .1f0
    internal_variance::Float32 = 0.40f0
    internal_clock_power::Float32 = 16f0
    internal_mobility_sigma::Float32 = .15f0
    internal_mobility_fade_start::Float32 = .8f0
    rotation_variance::Float32 = 6f0
    rotation_angle_cap::Float32 = 0.95f0 * Float32(pi)
    manifold_substep::Float32 = 0.005f0
    time_epsilon::Float32 = 1f-3
    inertia_regularisation::Float32 = 1f-5
    random_global_rotation::Bool = true
    minimum_fill_fraction::Float32 = .95f0
    max_load_attempts::Int = 4096
    atom_class_count::Int = 14
    mask_class::Int = 13
end


struct MolecularBridgeExample
    metadata::MolecularExample
    xt::Matrix{Float32}
    x1::Matrix{Float32}
    atom_class_t::Vector{UInt8}
    target_internal::Matrix{Float32}
    target_center::Matrix{Float32}
    target_rotation::Matrix{Float32}
    internal_variance::Vector{Float32}
    time::Float32
end


function _clean_molecular_latents(
    example::MolecularExample, coords::AbstractMatrix,
)
    nfragments = length(example.fragment_atom_indices)
    centers = Matrix{Float32}(undef, 3, nfragments)
    rotations = Array{Float32}(undef, 3, 3, nfragments)
    local_coords = similar(coords)
    for fragment in 1:nfragments
        indices = example.fragment_atom_indices[fragment]
        x = coords[:, indices]
        center = vec(sum(x; dims=2) ./ length(indices))
        template = example.template[:, indices]
        rotation = kabsch_rotation(template, x)
        centers[:, fragment] .= center
        rotations[:, :, fragment] .= rotation
        local_coords[:, indices] .= transpose(rotation) * (x .- center)
    end
    return centers, rotations, local_coords
end

function _project_fragment_internal!(
    values::AbstractMatrix,
    fragment_atom_indices::Vector{Vector{Int}},
)
    for indices in fragment_atom_indices
        values[:, indices] .-= sum(values[:, indices]; dims=2) ./ length(indices)
    end
    return values
end

function _structured_molecular_target!(
    dc::AbstractMatrix,
    dR::AbstractMatrix,
    dq::AbstractMatrix,
    raw::AbstractMatrix,
    residual::AbstractMatrix,
    fragment_atom_indices::Vector{Vector{Int}},
    regularisation::Float32,
)
    eye3 = Matrix{Float32}(I, 3, 3)
    for fragment in eachindex(fragment_atom_indices)
        indices = fragment_atom_indices[fragment]
        x = residual[:, indices]
        b = raw[:, indices]
        center = vec(sum(b; dims=2) ./ length(indices))
        z = b .- center
        inertia = zeros(Float32, 3, 3)
        rhs = zeros(Float32, 3)
        for atom in axes(x, 2)
            r = x[:, atom]
            inertia .+= dot(r, r) .* eye3 .- r * transpose(r)
            rhs .+= cross(r, z[:, atom])
        end
        inertia ./= length(indices)
        rhs ./= length(indices)
        rg2 = Float32(sum(abs2, x) / length(indices))
        omega = (
            inertia + regularisation * max(rg2, 1f0) * eye3
        ) \ rhs
        dc[:, fragment] .= center
        dR[:, fragment] .= omega
        for (local_atom, atom_index) in enumerate(indices)
            dq[:, atom_index] .= (
                z[:, local_atom] - cross(omega, x[:, local_atom])
            )
        end
    end
    atom_counts = length.(fragment_atom_indices)
    project_centers!(dc, atom_counts)
    return nothing
end

function _sample_atom_label_bridge(
    endpoint_class::Vector{UInt8},
    time::Float32,
    config::MolecularPrepConfig,
)
    process = Flowfusion.DistNoisyInterpolatingDiscreteFlow()
    # Flowfusion discrete states are one-based. Python receives zero-based
    # classes, with the last class reserved for the masked source token.
    x0 = DiscreteState(
        config.atom_class_count,
        fill(config.mask_class + 1, length(endpoint_class)),
    )
    x1 = DiscreteState(
        config.atom_class_count,
        Int.(endpoint_class) .+ 1,
    )
    xt = Flowfusion.bridge(process, x0, x1, time)
    return UInt8.(xt.state .- 1)
end

function _molecular_bridge_endpoints(
    example::MolecularExample,
    config::MolecularPrepConfig,
    rng::AbstractRNG,
)
    isfinite(config.internal_clock_power) && config.internal_clock_power > 0f0 ||
        throw(ArgumentError("internal clock power must be finite and positive"))
    isfinite(config.rotation_variance) && config.rotation_variance > 0 ||
        throw(ArgumentError("rotation variance must be a finite positive global constant"))
    _validate_internal_mobility(config)
    _validate_center_spread(config)
    x1 = copy(example.coords)
    project_global!(x1)
    if config.random_global_rotation
        x1 .= random_rotation(rng) * x1
    end
    c1, R1, q1 = _clean_molecular_latents(example, x1)
    nfragments = length(example.fragment_atom_indices)
    atom_counts = length.(example.fragment_atom_indices)

    x0 = config.source_sigma .* randn(rng, Float32, size(x1))
    project_global!(x0)
    c0 = Matrix{Float32}(undef, 3, nfragments)
    R0 = Array{Float32}(undef, 3, 3, nfragments)
    q0 = similar(x0)
    for fragment in 1:nfragments
        indices = example.fragment_atom_indices[fragment]
        center = vec(sum(x0[:, indices]; dims=2) ./ length(indices))
        rotation = (
            R1[:, :, fragment] *
            capped_rotation(rng, config.rotation_angle_cap)
        )
        c0[:, fragment] .= center
        R0[:, :, fragment] .= rotation
        q0[:, indices] .= transpose(rotation) * (x0[:, indices] .- center)
    end
    return (; x0, x1, c0, c1, R0, R1, q0, q1,
              nfragments, atom_counts)
end

function sample_molecular_bridge(
    example::MolecularExample,
    config::MolecularPrepConfig,
    rng::AbstractRNG=Random.default_rng();
    physical_time::Union{Nothing,Float32}=nothing,
)
    config.coordinate_process in CARTESIAN_BM_PROCESSES &&
        return sample_atom_bm_bridge(example, config, rng; physical_time)
    config.coordinate_process == "fragment-se3-plus-internal" ||
        throw(ArgumentError("unknown coordinate process"))
    (; x1, c0, c1, R0, R1, q0, q1,
       nfragments, atom_counts) = _molecular_bridge_endpoints(example, config, rng)

    t = (
        rand(rng, Float32) * (1f0 - 2f0 * config.time_epsilon) +
        config.time_epsilon
    )
    # Keep the default RNG draw/order unchanged. An explicit time is useful
    # for inspecting the exact training marginals at a fixed source/endpoint.
    if physical_time !== nothing
        0f0 <= physical_time <= 1f0 ||
            throw(ArgumentError("physical time must lie in [0, 1]"))
        t = physical_time
    end
    segment_s = segment_schedule(t)
    internal_s = internal_schedule(t, config.internal_clock_power)
    center_bridge_scale = sqrt(segment_s * (1f0 - segment_s))
    internal_bridge_scale = sqrt(internal_s * (1f0 - internal_s))

    center_noise = randn(rng, Float32, 3, nfragments)
    project_centers!(center_noise, atom_counts)
    center_scale = config.center_spread_sigma > 0 ?
        Float32(sqrt(_center_marginal_variance(t, config; s=segment_s))) :
        sqrt(config.center_variance) * center_bridge_scale
    ct = (
        (1f0 - segment_s) .* c0 .+
        segment_s .* c1 .+
        center_scale .* center_noise
    )
    project_centers!(ct, atom_counts)

    internal_noise = randn(rng, Float32, size(q0))
    _project_fragment_internal!(
        internal_noise, example.fragment_atom_indices,
    )
    qt = if config.internal_mobility_sigma > 0
        # Same fast mean relaxation, with persistent mean-reverting diffusion.
        loga = _internal_log_retention(t, config.internal_clock_power)
        Float32.(exp(loga) .* q0 .+ -expm1(loga) .* q1 .+
                  sqrt(_mobile_internal_variance(t, config)) .* internal_noise)
    else
        (
            (1f0 - internal_s) .* q0 .+
            internal_s .* q1 .+
            sqrt(config.internal_variance) *
            internal_bridge_scale .* internal_noise
        )
    end
    _project_fragment_internal!(qt, example.fragment_atom_indices)
    internal_variance = fill(config.internal_variance, nfragments)
    rotation_variance = fill(config.rotation_variance, nfragments)
    Rt = sample_rotation_bridges(
        rng,
        R0,
        R1,
        segment_s,
        rotation_variance,
        config.manifold_substep,
    )

    xt = similar(x1)
    residual = similar(x1)
    raw = similar(x1)
    for fragment in 1:nfragments
        indices = example.fragment_atom_indices[fragment]
        rotation = Rt[:, :, fragment]
        r = rotation * qt[:, indices]
        residual[:, indices] .= r
        xt[:, indices] .= ct[:, fragment] .+ r
        dc_raw = c1[:, fragment] - ct[:, fragment]
        dR_raw = log_so3(
            R1[:, :, fragment] * transpose(rotation)
        )
        dq_raw = rotation * (q1[:, indices] - qt[:, indices])
        for (local_atom, atom_index) in enumerate(indices)
            raw[:, atom_index] .= (
                dc_raw .+
                config.internal_clock_power .* dq_raw[:, local_atom] .+
                cross(dR_raw, r[:, local_atom])
            )
        end
    end
    project_global!(xt)

    target_center = zeros(Float32, 3, nfragments)
    target_rotation = zeros(Float32, 3, nfragments)
    target_internal = zeros(Float32, size(x1))
    _structured_molecular_target!(
        target_center,
        target_rotation,
        target_internal,
        raw,
        residual,
        example.fragment_atom_indices,
        config.inertia_regularisation,
    )
    target_internal ./= config.internal_clock_power
    atom_class_t = _sample_atom_label_bridge(
        example.atom_class, t, config,
    )
    return MolecularBridgeExample(
        example,
        xt,
        x1,
        atom_class_t,
        target_internal,
        target_center,
        target_rotation,
        internal_variance,
        t,
    )
end
