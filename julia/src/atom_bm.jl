const TIME_CHANGED_BM_PROCESS = "atom-brownian-kumaraswamy-v1"
const CARTESIAN_BM_PROCESSES = (TIME_CHANGED_BM_PROCESS,)

function atom_bm_options(path::AbstractString)
    isempty(path) && return (;)
    p = TOML.parsefile(path)
    options = (; coordinate_process=String(p["coordinate_process"]),
        atom_bm_variance=Float64(p["atom_bm_variance"]),
        atom_bm_clock_a=Float64(p["atom_bm_clock_a"]),
        atom_bm_clock_b=Float64(p["atom_bm_clock_b"]))
    _validate_atom_bm(MolecularPrepConfig(; options...))
    return options
end

function _validate_atom_bm(config)
    config.coordinate_process == TIME_CHANGED_BM_PROCESS || throw(ArgumentError("unsupported Brownian process"))
    all(x -> isfinite(x) && x > 0, (config.atom_bm_variance, config.atom_bm_clock_a, config.atom_bm_clock_b)) ||
        throw(ArgumentError("Brownian parameters must be finite and positive"))
end

function _bm_clock_log_retention(t, config)
    0 <= t <= 1 || throw(ArgumentError("time outside [0,1]"))
    t == 0 && return 0.
    t == 1 && return -Inf
    return config.atom_bm_clock_b * log(-expm1(config.atom_bm_clock_a * log(Float64(t))))
end

function _cartesian_bm_clock(t, config)
    log_a = _bm_clock_log_retention(t, config)
    return (exp(log_a), -expm1(log_a))
end

function _atom_bm_variance(t, config)
    a, u = _cartesian_bm_clock(t, config)
    return config.atom_bm_variance * a * u
end

"""Exact Gaussian transition coefficients before molecule-global centering."""
function _atom_bm_coefficients(start, stop, config)
    0 <= start <= stop <= 1 || throw(ArgumentError("invalid Brownian interval"))
    start == stop && return (1., 0., 0.)
    stop == 1 && return (0., 1., 0.)
    a0, a1 = _bm_clock_log_retention(start,config), _bm_clock_log_retention(stop,config)
    pull = -expm1(a1-a0)
    return (exp(a1-a0), pull, config.atom_bm_variance * exp(a1) * pull)
end

function _standard_bm_endpoints(example, config, rng)
    x1 = copy(example.coords)
    project_global!(x1)
    config.random_global_rotation && (x1 .= random_rotation(rng) * x1)
    x0 = config.source_sigma .* randn(rng, Float32, size(x1))
    project_global!(x0)
    return (; x0, x1, nfragments=length(example.fragment_atom_indices))
end

"""Atom-wise Brownian bridge in the selected scalar clock; no fragment dynamics."""
function sample_atom_bm_bridge(example, config, rng; physical_time=nothing)
    (; x0, x1, nfragments) = _standard_bm_endpoints(example, config, rng)
    t = physical_time === nothing ?
        rand(rng,Float32)*(1f0-2f0*config.time_epsilon)+config.time_epsilon : Float32(physical_time)
    0 <= t <= 1 || throw(ArgumentError("time outside [0,1]"))
    noise = randn(rng,Float32,size(x0))
    project_global!(noise)
    a, u = _cartesian_bm_clock(t, config)
    xt = Float32.(a.*x0 .+ u.*x1 .+ sqrt(_atom_bm_variance(t,config)).*noise)
    project_global!(xt)
    labels = _sample_atom_label_bridge(example.atom_class,t,config)
    # Unused structured fields are zero; the Cartesian loss directly computes x1-xt.
    return MolecularBridgeExample(example,xt,x1,labels,zeros(Float32,size(x1)),
        zeros(Float32,3,nfragments),zeros(Float32,3,nfragments),
        zeros(Float32,nfragments),t)
end

function sample_atom_bm_path(example, config, times; seed::Int=17)
    _validate_atom_bm(config)
    endpoints = _standard_bm_endpoints(example,config,MersenneTwister(seed))
    (; x0,x1) = endpoints
    positions = zeros(Float32,size(x1)...,length(times))
    positions[:,:,1] .= x0
    labels = zeros(UInt8,length(example.atom_class),length(times))
    rng, noise = MersenneTwister(seed+6), similar(x0)
    discrete = Flowfusion.DistNoisyInterpolatingDiscreteFlow()
    target = DiscreteState(config.atom_class_count,Int.(example.atom_class).+1)
    label_state = DiscreteState(config.atom_class_count,
        fill(config.mask_class+1,length(example.atom_class)))
    Random.seed!(seed+4)
    labels[:,1] .= UInt8(config.mask_class)
    for frame in 2:length(times)
        retain,pull,variance = _atom_bm_coefficients(times[frame-1],times[frame],config)
        randn!(rng,noise)
        project_global!(noise)
        positions[:,:,frame] .= retain.*positions[:,:,frame-1] .+ pull.*x1 .+ sqrt(variance).*noise
        project_global!(view(positions,:,:,frame))
        label_state = Flowfusion.bridge(discrete,label_state,target,
            Float64(times[frame-1]),Float64(times[frame]))
        labels[:,frame] .= UInt8.(label_state.state.-1)
    end
    # Computed for diagnostics only. They do not evolve as latent states.
    centers = zeros(Float32,3,endpoints.nfragments,length(times))
    residuals = similar(positions)
    for (fragment,indices) in enumerate(example.fragment_atom_indices)
        centers[:,fragment,:] .= dropdims(mean(positions[:,indices,:];dims=2);dims=2)
        residuals[:,indices,:] .= positions[:,indices,:] .- reshape(centers[:,fragment,:],3,1,:)
    end
    return (; positions,labels,centers,residuals,endpoints)
end
