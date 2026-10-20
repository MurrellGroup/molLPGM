struct MolecularTrainingBatch
    arrays::Dict{String,Any}
    source_names::Vector{String}
    document_lengths::Vector{Int}
    n_documents::Int
    n_atoms::Int
    n_fragments::Int
end


function _sample_molecular_documents(
    corpus::MolecularCorpus,
    indices::Vector{Int},
    config::MolecularPrepConfig,
    rng::AbstractRNG,
)
    selected = MolecularBridgeExample[]
    used = Set{Int}()
    tokens = 0
    attempts = 0
    minimum_tokens = ceil(Int, config.minimum_fill_fraction * config.token_capacity)
    0f0 < config.minimum_fill_fraction <= 1f0 ||
        throw(ArgumentError("minimum fill fraction must lie in (0, 1]"))
    while (
        attempts < config.max_load_attempts &&
        length(selected) < config.max_documents
    )
        attempts += 1
        conformer_index = rand(rng, indices)
        conformer_index in used && continue
        push!(used, conformer_index)
        atom_count = Int(
            corpus.conformer_offsets[conformer_index + 1] -
            corpus.conformer_offsets[conformer_index]
        )
        atom_count > config.token_capacity && continue
        if !isempty(selected) && tokens + atom_count > config.token_capacity
            continue
        end
        example = load_example(corpus, conformer_index)
        push!(selected, sample_molecular_bridge(example, config, rng))
        tokens += atom_count
        tokens >= minimum_tokens && break
    end
    isempty(selected) &&
        throw(ErrorException("could not sample a molecular document"))
    return selected
end

function make_molecular_training_batch(
    corpus::MolecularCorpus,
    indices::Vector{Int},
    config::MolecularPrepConfig,
    rng::AbstractRNG=Random.default_rng();
    selected_indices=nothing,
)
    examples = selected_indices === nothing ? _sample_molecular_documents(corpus, indices, config, rng) :
        [sample_molecular_bridge(load_example(corpus,i),config,rng) for i in selected_indices]
    token_capacity = config.token_capacity
    fragment_capacity = config.fragment_capacity
    total_fragments = sum(
        length(example.metadata.fragment_atom_indices) for example in examples
    )
    total_fragments <= fragment_capacity ||
        throw(ArgumentError(
            "fragment capacity $fragment_capacity is below $total_fragments"
        ))

    xt = zeros(Float32, 3, token_capacity)
    x1 = zeros(Float32, 3, token_capacity)
    target_internal = zeros(Float32, 3, token_capacity)
    atom_class = fill(UInt8(config.mask_class), token_capacity)
    target_atom_class = fill(UInt8(config.mask_class), token_capacity)
    atom_index = zeros(Int32, token_capacity)
    fragment_id = zeros(Int32, token_capacity)
    document_id = zeros(Int32, token_capacity)
    time_token = zeros(Float32, token_capacity)
    valid_atom = zeros(UInt8, token_capacity)

    target_center = zeros(Float32, 3, fragment_capacity)
    target_rotation = zeros(Float32, 3, fragment_capacity)
    fragment_length = zeros(Int32, fragment_capacity)
    fragment_document = zeros(Int32, fragment_capacity)
    fragment_atom_count = zeros(Int32, fragment_capacity)
    internal_variance = zeros(Float32, fragment_capacity)
    time_fragment = zeros(Float32, fragment_capacity)
    valid_fragment = zeros(UInt8, fragment_capacity)

    atom_offset = 0
    fragment_offset = 0
    source_names = String[]
    document_lengths = Int[]
    dataset = String(get(corpus.metadata, "dataset", ""))
    source_prefix = startswith(dataset, "QM9") ? "qm9" : "geom"
    for (document, example) in enumerate(examples)
        atom_count = size(example.xt, 2)
        fragment_count = length(example.metadata.fragment_atom_indices)
        atom_range = (atom_offset + 1):(atom_offset + atom_count)
        fragment_range = (
            fragment_offset + 1
        ):(fragment_offset + fragment_count)
        xt[:, atom_range] .= example.xt
        x1[:, atom_range] .= example.x1
        target_internal[:, atom_range] .= example.target_internal
        atom_class[atom_range] .= example.atom_class_t
        target_atom_class[atom_range] .= example.metadata.atom_class
        atom_index[atom_range] .= Int32.(0:(atom_count - 1))
        fragment_id[atom_range] .= (
            example.metadata.fragment_id .+ Int32(fragment_offset)
        )
        document_id[atom_range] .= Int32(document - 1)
        time_token[atom_range] .= example.time
        valid_atom[atom_range] .= UInt8(1)

        target_center[:, fragment_range] .= example.target_center
        target_rotation[:, fragment_range] .= example.target_rotation
        fragment_document[fragment_range] .= Int32(document - 1)
        time_fragment[fragment_range] .= example.time
        valid_fragment[fragment_range] .= UInt8(1)
        for local_fragment in 1:fragment_count
            count = length(
                example.metadata.fragment_atom_indices[local_fragment]
            )
            fragment_length[fragment_offset + local_fragment] = Int32(count)
            fragment_atom_count[fragment_offset + local_fragment] = Int32(count)
        end
        internal_variance[fragment_range] .= example.internal_variance
        push!(
            source_names,
            @sprintf(
                "%s_%d_%d_%d",
                source_prefix,
                example.metadata.split,
                example.metadata.molecule_id,
                example.metadata.conformer_id,
            ),
        )
        push!(document_lengths, atom_count)
        atom_offset += atom_count
        fragment_offset += fragment_count
    end
    if atom_offset < token_capacity
        document_id[(atom_offset + 1):token_capacity] .= Int32(length(examples))
    end

    arrays = Dict{String,Any}(
        "xt" => xt,
        "x1" => x1,
        "target_internal" => target_internal,
        "atom_class" => atom_class,
        "target_atom_class" => target_atom_class,
        "atom_index" => atom_index,
        "fragment_id" => fragment_id,
        "document_id" => document_id,
        "time_token" => time_token,
        "valid_atom" => valid_atom,
        "target_center" => target_center,
        "target_rotation" => target_rotation,
        "fragment_length" => fragment_length,
        "fragment_document" => fragment_document,
        "fragment_atom_count" => fragment_atom_count,
        "internal_variance" => internal_variance,
        "time_fragment" => time_fragment,
        "valid_fragment" => valid_fragment,
    )
    return MolecularTrainingBatch(
        arrays,
        source_names,
        document_lengths,
        length(examples),
        atom_offset,
        fragment_offset,
    )
end
