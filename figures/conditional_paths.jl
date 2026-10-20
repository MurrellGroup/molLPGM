# Usage: julia --project=julia figures/conditional_paths.jl --corpus=data/geom --output=artifacts/conditional-paths
# Uses the same conditional-path implementation as training.
using MolLPGMPrep, LinearAlgebra, Random, TOML, Printf, SHA
BLAS.set_num_threads(1)

function main()
    options = Dict(split(a[3:end], "="; limit=2) for a in ARGS)
    plan = TOML.parsefile(get(options, "selection", joinpath(@__DIR__, "reference/conditional-selection.toml")))
    output = abspath(options["output"])
    ispath(output) && error("Output already exists: $output")
    mkpath(output)
    domain = String(get(options, "domain", "geom-drugs"))
    prefix = domain == "qm9" ? "qm9" : "geom"
    corpus = load_corpus(abspath(options["corpus"]); domain)
    times = Float32.(range(0, 1; length=plan["steps"] + 1))
    # This observation grid is monotone under the actual Float32 training clock.
    @assert issorted(MolLPGMPrep.segment_schedule.(times))
    @assert issorted(MolLPGMPrep.internal_schedule.(times, 16f0))
    fragment = MolecularPrepConfig()
    bm = MolecularPrepConfig(; atom_bm_options(joinpath(@__DIR__, "../configs/processes/bm.toml"))...)
    records = Dict{String,Any}[]
    pairs = Dict{String,Any}[]
    limit = parse(Int, get(options, "limit", string(length(plan["sample_indices"]))))
    0 < limit <= length(plan["sample_indices"]) || error("Invalid limit")
    for row in 1:limit
        sample = plan["sample_indices"][row]
        index = plan["conformer_indices"][row]
        seed = plan["seed"] + 1000row
        folder = joinpath(output, @sprintf("sample_%06d", sample))
        receipt = joinpath(folder, "metadata.toml")
        ex = load_example(corpus, index)
        source = @sprintf("%s_%d_%d_%d", prefix, ex.split, ex.molecule_id, ex.conformer_id)
        @assert source == plan["source_names"][row]
        @assert ex.split == 2 && length(ex.fragment_atom_indices) >= 2
        paths = Dict{String,Any}()
        elapsed = @elapsed begin
            for (kind, config) in (("fragment", fragment), ("bm", bm))
                paths[kind] = sample_molecular_path(ex, config, times; seed)
            end
        end
        first_path, second_path = paths["fragment"], paths["bm"]
        @assert first_path.endpoints.x0 == second_path.endpoints.x0
        @assert first_path.endpoints.x1 == second_path.endpoints.x1
        @assert first_path.labels == second_path.labels
        @assert first_path.labels[:, end] == ex.atom_class
        mkpath(folder)
        write(joinpath(folder, "x0.f32"), first_path.endpoints.x0)
        write(joinpath(folder, "x1.f32"), first_path.endpoints.x1)
        record = Dict{String,Any}(
            "sample_index" => sample, "conformer_index" => index,
            "source_name" => source, "seed" => seed,
            "atom_count" => size(ex.coords, 2), "fragment_count" => length(ex.fragment_atom_indices),
            "fragment_ids" => Int.(ex.fragment_id), "atom_classes" => Int.(ex.atom_class),
            "frames" => length(times), "seconds" => elapsed,
            "paired_x0_bitwise_identical" => true, "paired_x1_bitwise_identical" => true,
            "paired_label_paths_identical" => true,
        )
        for kind in ("fragment", "bm")
            path = paths[kind]
            @assert all(isfinite, path.positions)
            first_error = maximum(abs.(path.positions[:, :, 1] .- path.endpoints.x0))
            last_error = maximum(abs.(path.positions[:, :, end] .- path.endpoints.x1))
            @assert max(first_error, last_error) < 3f-5
            localdir = joinpath(folder, kind)
            mkpath(localdir)
            # Keep the native endpoint reconstruction for the numerical audit.
            write(joinpath(localdir, "native_first.f32"), path.positions[:, :, 1])
            write(joinpath(localdir, "native_last.f32"), path.positions[:, :, end])
            # At the two boundaries, write the known analytical endpoints exactly.
            # The interior remains the unmodified production conditional path.
            path.positions[:, :, 1] .= path.endpoints.x0
            path.positions[:, :, end] .= path.endpoints.x1
            write(joinpath(localdir, "positions.f32"), path.positions)
            write(joinpath(localdir, "atom_labels.u8"), path.labels)
            record["$(kind)_native_x0_max_error_angstrom"] = Float64(first_error)
            record["$(kind)_native_x1_max_error_angstrom"] = Float64(last_error)
            metadata = Dict(
                "source_name"=>source, "atom_index"=>collect(1:size(ex.coords,2)),
                "segment_id"=>Int.(ex.fragment_id), "times"=>Float64.(times),
                "frame_count"=>length(times), "atom_count"=>size(ex.coords,2))
            open(io -> TOML.print(io, metadata; sorted=true), joinpath(localdir, "trajectory.toml"), "w")
            record["$(kind)_positions_sha256"] = bytes2hex(sha256(read(joinpath(localdir, "positions.f32"))))
        end
        @assert paths["fragment"].positions[:, :, 1] == paths["bm"].positions[:, :, 1]
        @assert paths["fragment"].positions[:, :, end] == paths["bm"].positions[:, :, end]
        open(io -> TOML.print(io, record; sorted=true), receipt, "w")
        push!(records, record)
        push!(pairs, Dict("sample_index"=>sample,
            "fragment"=>relpath(joinpath(folder,"fragment/trajectory.toml"),output),
            "bm"=>relpath(joinpath(folder,"bm/trajectory.toml"),output)))
        @printf("completed=%d/%d source=%s atoms=%d fragments=%d seconds=%.3f\n",
                row, limit, source, size(ex.coords, 2), length(ex.fragment_atom_indices), elapsed)
        flush(stdout)
    end
    manifest = Dict{String,Any}(
        "samples" => length(records), "frames" => length(times), "times" => Float64.(times),
        "source" => "$domain held-out corpus; cases specified in selection file",
        "x0" => "Shared centered iid Gaussian atom coordinates; standard deviation 3 angstrom",
        "x1" => "Shared dataset conformer with the same random global rotation",
        "path_function" => "MolLPGMPrep.sample_molecular_path",
        "fragment_process" => Dict(string(f)=>getfield(fragment, f) for f in fieldnames(MolecularPrepConfig)),
        "bm_process" => Dict(string(f)=>getfield(bm, f) for f in fieldnames(MolecularPrepConfig)),
        "observation_grid" => "900 uniform physical-time intervals; native Float32 clocks monotone on this grid",
        "gaussian_path" => "Sequential exact conditional transitions with fresh Brownian innovations",
        "rotation_path" => "Production 0.005 transformed-time anchors and Brownian-driven partial updates",
        "endpoint_representation" => "Analytical shared x0 and x1 written exactly at boundaries; native float32 reconstruction saved separately; interiors unchanged",
        "selections" => records,
    )
    filename = "manifest.toml"
    open(io -> TOML.print(io, Dict("pairs"=>pairs); sorted=true), joinpath(output, "pairs.toml"), "w")
    open(io -> TOML.print(io, manifest; sorted=true), joinpath(output, filename), "w")
end

main()
