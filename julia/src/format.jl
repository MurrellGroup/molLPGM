const RAW_DTYPES = Dict{DataType,String}(
    Float32 => "float32",
    Int32 => "int32",
    UInt8 => "uint8",
)

function _write_raw(path::String, array)
    open(path, "w") do io
        write(io, vec(Array(array)))
    end
end

function complete_batch_count(root::String)
    isdir(root) || return 0
    return count(
        name -> isfile(joinpath(root, name, "metadata.toml")),
        readdir(root),
    )
end

function write_molecular_batch_atomic(
    root::String,
    name::String,
    batch::MolecularTrainingBatch,
    config::MolecularPrepConfig;
    seed::Int,
    split::Symbol,
    domain::String="qm9",
)
    mkpath(root)
    final_dir = joinpath(root, name)
    ispath(final_dir) && throw(ArgumentError("batch already exists: $final_dir"))
    temporary_dir = joinpath(root, ".tmp-$(uuid4())")
    mkpath(temporary_dir)
    fields = Dict{String,Any}()
    try
        for field in sort!(collect(keys(batch.arrays)))
            array = batch.arrays[field]
            dtype = get(RAW_DTYPES, eltype(array), nothing)
            dtype === nothing &&
                throw(ArgumentError("unsupported dtype $(eltype(array))"))
            suffix = dtype == "float32" ? "f32" :
                     dtype == "int32" ? "i32" : "u8"
            filename = "$field.$suffix"
            _write_raw(joinpath(temporary_dir, filename), array)
            fields[field] = Dict(
                "file" => filename,
                "dtype" => dtype,
                "shape" => collect(size(array)),
                "order" => "F",
            )
        end
        metadata = Dict{String,Any}(
            "format_version" => 1,
            "domain" => domain,
            "seed" => seed,
            "split" => String(split),
            "n_documents" => batch.n_documents,
            "n_atoms" => batch.n_atoms,
            "n_fragments" => batch.n_fragments,
            "source_names" => batch.source_names,
            "document_lengths" => batch.document_lengths,
            "fields" => fields,
            "process" => Dict(
                "coordinate_process" => "fragment-se3-plus-internal",
                "fragment_rigid_body" => true,
                "schedule_profile" => "fragment-noise",
                "center_clock" => "smootherstep",
                "center_clock_power" => 1.0,
                "rotation_clock" => "smootherstep",
                "rotation_clock_power" => 1.0,
                "internal_clock" => "smootherstep_power",
                "internal_clock_power" => config.internal_clock_power,
                "target_parameterization" =>
                    "gauge_project(dc+dR_cross_r+$(config.internal_clock_power)dq);internal/$(config.internal_clock_power)",
                "fragmentation" => (
                    "BRICS-minimum-size-and-rank-coarsened"
                ),
                "token_capacity" => config.token_capacity,
                "fragment_capacity" => config.fragment_capacity,
                "source_sigma" => config.source_sigma,
                "center_variance" => config.center_variance,
                "internal_variance" => config.internal_variance,
                "rotation_noise_profile" => "constant-v1",
                "rotation_variance" => config.rotation_variance,
                "rotation_angle_cap" => config.rotation_angle_cap,
                "manifold_substep" => config.manifold_substep,
                "inertia_regularisation" => config.inertia_regularisation,
                "random_global_rotation" => config.random_global_rotation,
                "atom_class_count" => config.atom_class_count,
                "mask_class" => config.mask_class,
                "atom_label_process" => (
                    "Flowfusion.DistNoisyInterpolatingDiscreteFlow"
                ),
                "atom_label_D1" => "Beta(2,2)",
                "atom_label_D2" => "Beta(2,2)",
                "atom_label_uniform_weight" => 0.2,
            ),
        )
        if config.internal_mobility_sigma > 0
            metadata["process"]["internal_mobility_sigma"] = config.internal_mobility_sigma
            metadata["process"]["internal_mobility_fade_start"] = config.internal_mobility_fade_start
            metadata["process"]["internal_noise_profile"] = "mean-reverting-mobility-v1"
        end
        if config.center_spread_sigma > 0
            metadata["process"]["center_spread_sigma"] = config.center_spread_sigma
            metadata["process"]["center_spread_timescale"] = config.center_spread_timescale
            metadata["process"]["center_noise_profile"] = "early-diffusion-v1"
        end
        if config.coordinate_process in CARTESIAN_BM_PROCESSES
            _validate_atom_bm(config)
            common = metadata["process"]
            process = Dict{String,Any}(k => common[k] for k in (
                "fragmentation", "token_capacity", "fragment_capacity", "source_sigma",
                "random_global_rotation", "atom_class_count", "mask_class", "atom_label_process",
                "atom_label_D1", "atom_label_D2", "atom_label_uniform_weight"))
            merge!(process, Dict("coordinate_process" => config.coordinate_process,
                "fragment_rigid_body" => false, "atom_clock" => "kumaraswamy",
                "schedule_profile" => "time-changed-brownian",
                "atom_bm_clock_a" => config.atom_bm_clock_a,
                "atom_bm_clock_b" => config.atom_bm_clock_b,
                "atom_bm_variance" => config.atom_bm_variance,
                "target_parameterization" => "cartesian-displacement-x1-minus-xt-angstrom",
                "noise_projection" => "molecule-global centering only",
                "structured_targets" => "unused"))
            metadata["process"] = process
        end
        open(joinpath(temporary_dir, "metadata.toml"), "w") do io
            TOML.print(io, metadata; sorted=true)
        end
        mv(temporary_dir, final_dir)
    catch
        isdir(temporary_dir) &&
            rm(temporary_dir; recursive=true, force=true)
        rethrow()
    end
    return final_dir
end
