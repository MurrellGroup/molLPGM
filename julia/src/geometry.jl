hat(v::AbstractVector) = Float32[
    0       -v[3]    v[2]
    v[3]     0      -v[1]
   -v[2]     v[1]    0
]

function exp_so3(v::AbstractVector)
    theta2 = dot(v, v)
    K = hat(v)
    if theta2 < 1f-10
        return Matrix{Float32}(I, 3, 3) + K + 0.5f0 * (K * K)
    end
    theta = sqrt(theta2)
    return Matrix{Float32}(I, 3, 3) +
           (sin(theta) / theta) * K +
           ((1f0 - cos(theta)) / theta2) * (K * K)
end

function _rotation_quaternion(R::AbstractMatrix)
    trace_r = tr(R)
    if trace_r > 0
        s = 2f0 * sqrt(max(1f0 + trace_r, 0f0))
        q = Float32[s / 4f0, (R[3,2] - R[2,3]) / s,
                    (R[1,3] - R[3,1]) / s, (R[2,1] - R[1,2]) / s]
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = 2f0 * sqrt(max(1f0 + R[1,1] - R[2,2] - R[3,3], 0f0))
        q = Float32[(R[3,2] - R[2,3]) / s, s / 4f0,
                    (R[1,2] + R[2,1]) / s, (R[1,3] + R[3,1]) / s]
    elseif R[2,2] > R[3,3]
        s = 2f0 * sqrt(max(1f0 + R[2,2] - R[1,1] - R[3,3], 0f0))
        q = Float32[(R[1,3] - R[3,1]) / s, (R[1,2] + R[2,1]) / s,
                    s / 4f0, (R[2,3] + R[3,2]) / s]
    else
        s = 2f0 * sqrt(max(1f0 + R[3,3] - R[1,1] - R[2,2], 0f0))
        q = Float32[(R[2,1] - R[1,2]) / s, (R[1,3] + R[3,1]) / s,
                    (R[2,3] + R[3,2]) / s, s / 4f0]
    end
    q ./= norm(q)
    q[1] < 0 && (q .*= -1f0)
    return q
end

function log_so3(R::AbstractMatrix)
    q = _rotation_quaternion(R)
    nv = norm(view(q, 2:4))
    nv < 1f-7 && return 2f0 .* q[2:4]
    theta = 2f0 * atan(nv, clamp(q[1], 0f0, 1f0))
    return Float32(theta) .* q[2:4] ./ nv
end

function quaternion_rotation(q::AbstractVector)
    w, x, y, z = q
    return Float32[
        1-2(y*y+z*z)  2(x*y-w*z)    2(x*z+w*y)
        2(x*y+w*z)    1-2(x*x+z*z)  2(y*z-w*x)
        2(x*z-w*y)    2(y*z+w*x)    1-2(x*x+y*y)
    ]
end

function random_rotation(rng::AbstractRNG)
    q = randn(rng, Float32, 4)
    q ./= norm(q)
    return quaternion_rotation(q)
end

function capped_rotation(rng::AbstractRNG, cap::Float32)
    0f0 < cap < Float32(pi) ||
        throw(ArgumentError("rotation cap must lie in (0, π)"))
    while true
        q = randn(rng, Float32, 4)
        q ./= norm(q)
        q[1] < 0 && (q .*= -1f0)
        2f0 * acos(clamp(q[1], 0f0, 1f0)) <= cap &&
            return quaternion_rotation(q)
    end
end

function kabsch_rotation(template::AbstractMatrix, coords::AbstractMatrix)
    size(template) == size(coords) ||
        throw(DimensionMismatch("Kabsch inputs differ"))
    t = template .- sum(template; dims=2) ./ size(template, 2)
    x = coords .- sum(coords; dims=2) ./ size(coords, 2)
    decomp = svd(x * transpose(t))
    correction = Matrix{Float32}(I, 3, 3)
    correction[3,3] = det(decomp.U * decomp.Vt) < 0 ? -1f0 : 1f0
    return Matrix{Float32}(decomp.U * correction * decomp.Vt)
end

"""Quintic smootherstep clock for fragment centers and rotations."""
function segment_schedule(t::Float32)
    time = clamp(t, 0f0, 1f0)
    return clamp(
        time^3 * (10f0 - 15f0 * time + 6f0 * time^2),
        0f0,
        1f0,
    )
end

"""Internal clock Q=1-(1-S)^p, with p times the normalized rate of S."""
function internal_schedule(t::Float32, power::Float32=2f0)
    remaining = 1f0 - segment_schedule(t)
    return 1f0 - (power == 2f0 ? remaining^2 : remaining^power)
end

function project_global!(coords::AbstractMatrix)
    coords .-= sum(coords; dims=2) ./ size(coords, 2)
    return coords
end

function project_centers!(values::AbstractMatrix, atom_counts::AbstractVector)
    total = sum(atom_counts)
    shift = vec(sum(values .* reshape(atom_counts, 1, :); dims=2) ./ total)
    values .-= shift
    return values
end

"""One production noise-and-pull step; noise contains standard normal draws."""
function _rotation_bridge_update!(rotations, targets, start_s, ds, variances, noise)
    alpha = ds / (1f0 - start_s)
    for fragment in axes(rotations, 3)
        angular_noise = sqrt(max(variances[fragment] * ds / 2f0, 0f0)) .*
                        view(noise, :, fragment)
        noisy = exp_so3(angular_noise) * view(rotations, :, :, fragment)
        pull = log_so3(view(targets, :, :, fragment) * transpose(noisy))
        view(rotations, :, :, fragment) .= exp_so3(alpha .* pull) * noisy
    end
    return rotations
end

function _advance_rotation_bridges!(
    rng::AbstractRNG,
    rotations::AbstractArray{Float32,3},
    targets::AbstractArray{Float32,3},
    start_s::Float32,
    stop_s::Float32,
    variances::AbstractVector{Float32},
    max_substep::Float32,
)
    size(rotations) == size(targets) ||
        throw(DimensionMismatch("rotation stacks differ"))
    size(rotations, 3) == length(variances) ||
        throw(DimensionMismatch("rotation variance count differs"))
    0f0 <= start_s <= stop_s <= 1f0 ||
        throw(ArgumentError("rotation step must lie inside [0, 1]"))
    max_substep > 0f0 ||
        throw(ArgumentError("rotation substep must be positive"))
    noise = Matrix{Float32}(undef, 3, size(rotations, 3))
    current = start_s
    while current < stop_s - 1f-7
        ds = min(max_substep, stop_s - current)
        randn!(rng, noise)
        _rotation_bridge_update!(rotations, targets, current, ds, variances, noise)
        current += ds
    end
    stop_s >= 1f0 - 1f-7 && (rotations .= targets)
    return rotations
end

function sample_rotation_bridges(
    rng::AbstractRNG,
    R0::AbstractArray{Float32,3},
    R1::AbstractArray{Float32,3},
    transformed_time::Float32,
    variances::AbstractVector{Float32},
    max_substep::Float32,
)
    rotations = copy(R0)
    return _advance_rotation_bridges!(
        rng, rotations, R1, 0f0, transformed_time, variances, max_substep,
    )
end
