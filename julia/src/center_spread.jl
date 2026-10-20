"""Optional early centroid diffusion; keeps the source, mean and endpoint pull."""
function _validate_center_spread(config)
    isfinite(config.center_spread_sigma) && config.center_spread_sigma >= 0 ||
        throw(ArgumentError("center spread sigma must be finite and nonnegative"))
    isfinite(config.center_spread_timescale) && config.center_spread_timescale > 0 ||
        throw(ArgumentError("center spread timescale must be finite and positive"))
end

_center_spread_h(t, tau) = -expm1(-(Float64(t) / Float64(tau))^2)

"""Unprojected variance at one time, in Å². Uses the production center clock."""
function _center_marginal_variance(t, config; s=segment_schedule(Float32(t)))
    0 <= t <= 1 || throw(ArgumentError("invalid physical time"))
    a = 1 - Float64(s)
    return Float64(config.center_variance) * Float64(s) * a +
        Float64(config.center_spread_sigma)^2 * a^2 *
        _center_spread_h(t, config.center_spread_timescale)
end

"""Extra innovation variance over an interval; independent of base bridge noise."""
function _center_spread_innovation(start, stop, config;
                                   stop_s=segment_schedule(Float32(stop)))
    0 <= start <= stop <= 1 || throw(ArgumentError("invalid bridge interval"))
    # Avoid subtracting two h values rounded to one in the late tail.
    u, v = Float64(start)/config.center_spread_timescale, Float64(stop)/config.center_spread_timescale
    dh = exp(-u^2) * -expm1(-(v^2-u^2))
    return Float64(config.center_spread_sigma)^2 * (1-Float64(stop_s))^2 * dh
end
