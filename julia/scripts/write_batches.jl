using MolLPGMPrep
using LinearAlgebra
using Printf
using Random

# Each producer performs many small 3×3 operations. Process-level sharding is
# faster and avoids oversubscribing the host with one BLAS pool per writer.
BLAS.set_num_threads(1)

function arg(name::String, default::String)
    prefix = "--$name="
    for value in ARGS
        startswith(value, prefix) &&
            return value[(length(prefix) + 1):end]
    end
    return default
end

arg_int(name, default) = parse(Int, arg(name, string(default)))
arg_float(name, default) = parse(Float32, arg(name, string(default)))
arg_bool(name, default) =
    lowercase(arg(name, string(default))) in ("1", "true", "yes")

function main()
    domain = arg("domain", "qm9")
    domain in ("qm9", "geom-drugs") ||
        throw(ArgumentError("--domain must be qm9 or geom-drugs"))
    is_geom = domain == "geom-drugs"
    corpus_dir = abspath(arg("corpus-dir", joinpath(
        @__DIR__, "..", "..", "..",
        is_geom ? "geom_drugs_preprocessed" : "qm9_fragmented_preprocessed",
    )))
    out_dir = abspath(arg(
        "out-dir", joinpath(pwd(), is_geom ? "geom_batches" : "qm9_batches")
    ))
    batches = arg_int("batches", 16)
    continuous = arg_bool("continuous", false)
    queue_limit = arg_int("queue-limit", 0)
    poll_seconds = Float64(arg_float("queue-poll-seconds", 0.05))
    seed = arg_int("seed", 2718)
    shard_index = arg_int("shard-index", 1)
    shards = arg_int("shards", 1)
    1 <= shard_index <= shards ||
        throw(ArgumentError("--shard-index must lie in 1:--shards"))
    split = Symbol(arg("split", "train"))
    config = MolecularPrepConfig(;
        atom_bm_options(arg("atom-bm-profile", ""))...,
        token_capacity=arg_int("token-capacity", 4096),
        fragment_capacity=arg_int(
            "fragment-capacity", 1365,
        ),
        max_documents=arg_int("max-documents", 512),
        source_sigma=arg_float("source-sigma", 3),
        center_variance=arg_float("center-variance", 100 / 9),
        center_spread_sigma=arg_float("center-spread-sigma", 3),
        center_spread_timescale=arg_float("center-spread-timescale", .1),
        internal_variance=arg_float("internal-variance", 0.40),
        internal_clock_power=arg_float("internal-clock-power", 16),
        internal_mobility_sigma=arg_float("internal-mobility-sigma", .15),
        internal_mobility_fade_start=arg_float("internal-mobility-fade-start", .8),
        rotation_variance=arg_float("rotation-variance", 6),
        manifold_substep=arg_float("manifold-substep", 0.005),
        random_global_rotation=arg_bool("random-global-rotation", true),
        minimum_fill_fraction=arg_float(
            "minimum-fill-fraction", 0.95,
        ),
    )
    corpus = load_corpus(corpus_dir; domain)
    indices = split_indices(corpus, split)
    indices = indices[shard_index:shards:end]
    isempty(indices) && error("shard contains no conformers")
    local_seed = seed + 1_000_000 * (shard_index - 1)
    # Flowfusion's bridge uses the task-local default RNG internally, while
    # the continuous path receives `rng` explicitly. Seed both streams.
    Random.seed!(local_seed)
    rng = MersenneTwister(local_seed)
    mkpath(out_dir)
    @printf(
        "conformers=%d corpus=%s out_dir=%s split=%s shard=%d/%d\n",
        length(indices), corpus_dir, out_dir, split, shard_index, shards,
    )
    index = 1
    prefix = @sprintf("batch_s%04d_", shard_index)
    for name in readdir(out_dir)
        startswith(name, prefix) || continue
        isfile(joinpath(out_dir, name, "metadata.toml")) || continue
        existing = tryparse(Int, name[(length(prefix) + 1):end])
        existing === nothing || (index = max(index, existing + 1))
    end
    while continuous || index <= batches
        while (
            queue_limit > 0 &&
            complete_batch_count(out_dir) >= queue_limit
        )
            sleep(poll_seconds)
        end
        start = time()
        batch = make_molecular_training_batch(corpus, indices, config, rng)
        name = @sprintf("batch_s%04d_%09d", shard_index, index)
        path = write_molecular_batch_atomic(
            out_dir,
            name,
            batch,
            config;
            seed=local_seed,
            split,
            domain,
        )
        @printf(
            "wrote=%s documents=%d atoms=%d fragments=%d elapsed_s=%.3f\n",
            path,
            batch.n_documents,
            batch.n_atoms,
            batch.n_fragments,
            time() - start,
        )
        flush(stdout)
        index += 1
    end
end

main()
