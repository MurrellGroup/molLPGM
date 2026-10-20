"""Validate the mobile internal bridge parameters."""
function _validate_internal_mobility(config)
    sigma = config.internal_mobility_sigma
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("mobility sigma must be finite and nonnegative"))
    if sigma > 0
        config.internal_clock_power >= 1 || throw(ArgumentError("mobile bridge requires power >= 1"))
        .5 <= config.internal_mobility_fade_start < 1 ||
            throw(ArgumentError("mobility fade must start in [0.5, 1)"))
    end
end

# 1-S(t)=(1-t)^3(1+3t+6t^2). Never subtract a rounded clock from one:
# the power-16 clock rounds to one long before the physical endpoint.
function _internal_log_retention(t, power)
    time = Float64(t)
    0 <= time <= 1 || throw(ArgumentError("physical time must lie in [0, 1]"))
    time == 1 && return -Inf
    return Float64(power) * (3log1p(-time) + log1p(3time + 6time^2))
end

function _internal_mobility_fade(t, start)
    t <= start && return 1.0
    u = (Float64(t) - Float64(start)) / (1 - Float64(start))
    return (1-u)^3 * (1+3u+6u^2)
end

"""Variance before the per-fragment mean-free projection, in square angstroms."""
function _mobile_internal_variance(t, config)
    loga = _internal_log_retention(t, config.internal_clock_power)
    a = exp(loga)
    return Float64(config.internal_variance) * a * -expm1(loga) +
           Float64(config.internal_mobility_sigma)^2 * -expm1(2loga) *
           _internal_mobility_fade(t, config.internal_mobility_fade_start)
end

"""Exact linear-SDE transition: retention, endpoint pull, fresh-noise variance."""
function _mobile_internal_coefficients(start, stop, config)
    0 <= start <= stop <= 1 || throw(ArgumentError("invalid bridge times"))
    stop == 1 && return (0.0, 1.0, 0.0)
    start == stop && return (1.0, 0.0, 0.0)
    logratio = _internal_log_retention(stop, config.internal_clock_power) -
               _internal_log_retention(start, config.internal_clock_power)
    retention = exp(logratio)
    variance = _mobile_internal_variance(stop, config) -
               retention^2 * _mobile_internal_variance(start, config)
    variance >= -1e-12 || error("negative mobile bridge transition variance: $variance")
    return retention, -expm1(logratio), max(variance, 0.0)
end

function _mobile_internal_transition!(state, target, start, stop, config, noise)
    retention, pull, variance = _mobile_internal_coefficients(start, stop, config)
    state .= retention .* state .+ pull .* target .+ sqrt(variance) .* noise
    return state
end
