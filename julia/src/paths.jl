"""Exact Brownian-bridge transition in transformed time, with fresh noise."""
function _euclidean_bridge_transition!(state, target, start_s, stop_s, variance, noise)
    0 <= start_s <= stop_s <= 1 || throw(ArgumentError("invalid bridge times"))
    if stop_s == 1
        state .= target
        return state
    end
    stop_s == start_s && return state
    ds = stop_s - start_s
    alpha = ds / (1 - start_s)
    scale = sqrt(variance * ds * (1 - stop_s) / (1 - start_s))
    state .= (1 - alpha) .* state .+ alpha .* target .+ scale .* noise
    return state
end

"""
Brownian-driven path of the *production* rotation discretization.

The training marginal executes full max_substep noise-and-pull steps from
zero, followed by a partial step to the requested time. Retain those exact
anchors. Inside a partial step, accumulate Brownian increments from its
anchor, then apply the very same production update. Thus each observation
has the production marginal while different observations share a Brownian
path, rather than a single standard normal scaled by sqrt(elapsed).
This preserves the training discretization, including its numerical error;
it does not silently replace it with a finer sequential rotation solver.
"""
function _rotation_bridge_path(rng, R0, R1, clock, variances, max_substep)
    issorted(clock) && first(clock) == 0 && last(clock) == 1 ||
        throw(ArgumentError("path clock must increase from zero to one"))
    result = Array{Float32}(undef, size(R0)..., length(clock))
    anchor = copy(R0)
    brownian = zeros(Float32, 3, size(R0, 3))
    noise = similar(brownian)
    current = 0f0
    anchor_s = 0f0
    boundary = min(max_substep, 1f0)
    result[:, :, :, 1] .= R0
    for frame in 2:length(clock)
        stop_s = clock[frame]
        # Cross actual training substep boundaries, even if no display frame
        # lands on one. Random increments are fresh; no RNG is reset here.
        while current < stop_s
            next_s = min(stop_s, boundary)
            randn!(rng, noise)
            brownian .+= sqrt(next_s - current) .* noise
            current = next_s
            if current == boundary
                ds = boundary - anchor_s
                _rotation_bridge_update!(anchor, R1, anchor_s, ds, variances,
                                         brownian ./ sqrt(ds))
                anchor_s = boundary
                boundary = min(anchor_s + max_substep, 1f0)
                fill!(brownian, 0f0)
            end
        end
        rotation = copy(anchor)
        ds = stop_s - anchor_s
        # This tolerance and endpoint snap are the production rule.
        if ds > 1f-7
            _rotation_bridge_update!(rotation, R1, anchor_s, ds, variances,
                                     brownian ./ sqrt(ds))
        end
        stop_s >= 1f0 - 1f-7 && (rotation .= R1)
        result[:, :, :, frame] .= rotation
    end
    return result
end

"""
Sample a temporally consistent conditional path of the training process.

Centers and internals use exact Gaussian bridge transitions; rotations keep
the production substep discretization with Brownian increments; atom labels
use Flowfusion's exact two-time transition. Independent RNG streams prevent
the changed internal clock from changing centers, rotations or labels.
"""
function sample_molecular_path(example, config, times; seed::Int=17)
    length(times) >= 2 && first(times) == 0 && last(times) == 1 &&
        all(diff(times) .> 0) || throw(ArgumentError("times must increase from 0 to 1"))
    config.coordinate_process in CARTESIAN_BM_PROCESSES &&
        return sample_atom_bm_path(example, config, times; seed)
    config.coordinate_process == "fragment-se3-plus-internal" ||
        throw(ArgumentError("unknown coordinate process"))
    endpoints = _molecular_bridge_endpoints(example, config, MersenneTwister(seed))
    (; x1, c0, c1, R0, R1, q0, q1, atom_counts) = endpoints
    rigid_clock = segment_schedule.(Float32.(times))
    internal_clock = internal_schedule.(Float32.(times), config.internal_clock_power)
    # Quintic float32 cancellation near one can make the production clock
    # decrease slightly. Reject that grid rather than modify the process.
    issorted(rigid_clock) && issorted(internal_clock) ||
        throw(ArgumentError("production float32 clocks are not monotone on this grid"))
    variances = fill(config.rotation_variance, length(atom_counts))
    rotations = _rotation_bridge_path(MersenneTwister(seed+1), R0, R1,
                                     rigid_clock, variances, config.manifold_substep)
    centers = zeros(Float32, size(c0)..., length(times))
    internals = zeros(Float32, size(q0)..., length(times))
    positions = zeros(Float32, size(x1)..., length(times))
    labels = zeros(UInt8, length(example.atom_class), length(times))
    centers[:, :, 1] .= project_centers!(copy(c0), atom_counts)
    internals[:, :, 1] .= _project_fragment_internal!(copy(q0), example.fragment_atom_indices)
    center_rng, internal_rng = MersenneTwister(seed+2), MersenneTwister(seed+3)
    center_noise, internal_noise = similar(c0), similar(q0)
    spread_rng = MersenneTwister(seed+5)
    spread_noise = similar(c0)
    discrete = Flowfusion.DistNoisyInterpolatingDiscreteFlow()
    target = DiscreteState(config.atom_class_count, Int.(example.atom_class) .+ 1)
    label_state = DiscreteState(config.atom_class_count,
                               fill(config.mask_class+1, length(example.atom_class)))
    # Flowfusion uses Julia's task-local RNG. Seed once for the whole path.
    Random.seed!(seed+4)
    labels[:, 1] .= UInt8(config.mask_class)
    for frame in 2:length(times)
        c = copy(centers[:, :, frame-1])
        randn!(center_rng, center_noise)
        project_centers!(center_noise, atom_counts)
        _euclidean_bridge_transition!(c, c1, rigid_clock[frame-1],
                                     rigid_clock[frame], config.center_variance, center_noise)
        if config.center_spread_sigma > 0
            randn!(spread_rng, spread_noise)
            project_centers!(spread_noise, atom_counts)
            variance = _center_spread_innovation(times[frame-1], times[frame], config;
                                                stop_s=rigid_clock[frame])
            c .+= Float32(sqrt(variance)) .* spread_noise
        end
        project_centers!(c, atom_counts)
        centers[:, :, frame] .= c
        q = copy(internals[:, :, frame-1])
        randn!(internal_rng, internal_noise)
        _project_fragment_internal!(internal_noise, example.fragment_atom_indices)
        if config.internal_mobility_sigma > 0
            _mobile_internal_transition!(q, q1, times[frame-1], times[frame], config, internal_noise)
        else
            _euclidean_bridge_transition!(q, q1, internal_clock[frame-1],
                                         internal_clock[frame], config.internal_variance, internal_noise)
        end
        _project_fragment_internal!(q, example.fragment_atom_indices)
        internals[:, :, frame] .= q
        label_state = Flowfusion.bridge(discrete, label_state, target,
                                       Float64(times[frame-1]), Float64(times[frame]))
        labels[:, frame] .= UInt8.(label_state.state .- 1)
    end
    for frame in eachindex(times)
        for (fragment, indices) in enumerate(example.fragment_atom_indices)
            positions[:, indices, frame] .= centers[:, fragment, frame] .+
                rotations[:, :, fragment, frame] * internals[:, indices, frame]
        end
        project_global!(view(positions, :, :, frame))
    end
    return (; positions, labels, centers, internals, rotations,
              rigid_clock, internal_clock, endpoints)
end
